package com.lis.distributed.thread.pool;

import com.lis.distributed.thread.pool.client.ThreadPoolClient;
import com.lis.distributed.thread.pool.func.FuncUtils;
import com.lis.distributed.thread.pool.func.SerializableBiConsumer;
import com.lis.distributed.thread.pool.func.SerializableBiFunction;
import com.lis.distributed.thread.pool.func.SerializableConsumer;
import com.lis.distributed.thread.pool.func.SerializableFunction;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.EOFException;
import java.io.ObjectInputFilter;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.ServerSocket;
import java.time.Duration;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class DistributedThreadPoolTest {
    @Test
    void repositoryCoversRegistrationCompletionFailureCloseAndUnknownIds() {
        var repository = new DataRepository();

        var first = repository.<String>register();
        var second = repository.<Integer>register();

        assertEquals(0, first.id());
        assertEquals(1, second.id());
        assertEquals(2, repository.pendingCount());

        repository.complete(first.id(), "done");
        assertEquals("done", first.future().join());
        assertEquals(1, repository.pendingCount());

        var expected = new IllegalStateException("failed");
        repository.fail(second.id(), expected);
        var failure = assertThrows(CompletionException.class, second.future()::join);
        assertSame(expected, failure.getCause());
        assertEquals(0, repository.pendingCount());

        repository.complete(999, "ignored");
        repository.fail(999, new IOException("ignored"));
        assertEquals(0, repository.pendingCount());
        assertThrows(NullPointerException.class, () -> repository.fail(1, null));
        assertThrows(NullPointerException.class, () -> repository.failAll(null));

        var third = repository.<String>register();
        var fourth = repository.<String>register();
        var closeFailure = new IOException("close all");
        repository.failAll(closeFailure);
        assertTrue(third.future().isCompletedExceptionally());
        assertTrue(fourth.future().isCompletedExceptionally());
        assertEquals(0, repository.pendingCount());

        var pendingAtClose = repository.<String>register();
        repository.close();
        repository.close();
        assertTrue(pendingAtClose.future().isCompletedExceptionally());
        assertThrows(RejectedExecutionException.class, repository::register);
    }

    @Test
    void wireMessagesValidateSerializeFailuresAndExposeRemoteDetails() {
        assertThrows(IllegalArgumentException.class, () -> new WireMessage.Invocation(-1, () -> 1));
        assertThrows(NullPointerException.class, () -> new WireMessage.Invocation(0, null));
        var invocation = new WireMessage.Invocation(3, () -> 7);
        assertEquals(3, invocation.requestId());

        assertThrows(IllegalArgumentException.class,
                () -> new WireMessage.Command(-1, (TaskMessage<Object>) ignored -> {}));
        assertThrows(NullPointerException.class, () -> new WireMessage.Command(0, null));
        var command = new WireMessage.Command(4, (TaskMessage<Object>) ignored -> {});
        assertEquals(4, command.requestId());

        assertThrows(IllegalArgumentException.class,
                () -> new WireMessage.Response(-1, "x", null));
        assertThrows(IllegalArgumentException.class,
                () -> new WireMessage.Response(1, new Object(), null));

        var success = WireMessage.Response.success(5, "value");
        assertEquals(5, success.requestId());
        assertEquals("value", success.value());
        assertNull(success.failure());

        var original = new IllegalStateException("remote boom");
        var failed = WireMessage.Response.failure(6, original);
        assertNull(failed.value());
        assertNotNull(failed.failure());
        assertEquals(IllegalStateException.class.getName(), failed.failure().type());
        assertEquals("remote boom", failed.failure().message());
        assertTrue(failed.failure().stackTrace().contains("remote boom"));

        var remote = failed.failure().toException();
        assertEquals(IllegalStateException.class.getName(), remote.remoteType());
        assertTrue(remote.remoteStackTrace().contains("remote boom"));
        assertTrue(remote.getMessage().contains("remote boom"));

        var withoutMessage = new RemoteExecutionException("Type", null, "stack");
        assertEquals("Type", withoutMessage.getMessage());
        assertEquals("Type", withoutMessage.remoteType());
        assertEquals("stack", withoutMessage.remoteStackTrace());

        assertThrows(NullPointerException.class, () -> WireMessage.RemoteFailure.from(null));
        assertThrows(NullPointerException.class,
                () -> new WireMessage.RemoteFailure(null, "message", "stack"));
        assertThrows(NullPointerException.class,
                () -> new WireMessage.RemoteFailure("type", "message", null));

        assertThrows(IllegalArgumentException.class, () -> new WireMessage.Registration(0));
        assertThrows(IllegalArgumentException.class, () -> new WireMessage.Registration(-1));
        assertEquals(9, new WireMessage.Registration(9).clientId());
    }

    @Test
    void transportOptionsValidationAndCopiesAreCovered() {
        var defaults = SocketAccessor.Options.defaults();

        assertEquals(64 * 1024, defaults.ioBufferBytes());
        assertEquals(64, defaults.writerBatchSize());
        assertTrue(defaults.tcpNoDelay());
        assertTrue(defaults.virtualReader());
        assertTrue(defaults.virtualWriter());
        assertNull(defaults.inputFilter());

        assertThrows(IllegalArgumentException.class,
                () -> new SocketAccessor.Options(1023, 1, true, true, true, null));
        assertThrows(IllegalArgumentException.class,
                () -> new SocketAccessor.Options(1024, 0, true, true, true, null));

        ObjectInputFilter filter = info -> ObjectInputFilter.Status.UNDECIDED;
        var changed = defaults
                .withIoBufferBytes(2048)
                .withWriterBatchSize(1)
                .withTcpNoDelay(false)
                .withVirtualReader(false)
                .withVirtualWriter(false)
                .withInputFilter(filter);

        assertEquals(2048, changed.ioBufferBytes());
        assertEquals(1, changed.writerBatchSize());
        assertFalse(changed.tcpNoDelay());
        assertFalse(changed.virtualReader());
        assertFalse(changed.virtualWriter());
        assertSame(filter, changed.inputFilter());
    }

    @Test
    void utilityBindersAndNumberGeneratorAreCovered() throws Exception {
        SerializableFunction<Integer, Integer> plusOne = value -> value + 1;
        assertEquals(5, FuncUtils.bind(plusOne, 4).get());

        SerializableBiFunction<Integer, Integer, Integer> add = Integer::sum;
        assertEquals(9, FuncUtils.bind(add, 4).apply(5));

        var holder = new int[1];
        SerializableBiConsumer<int[], Integer> store = (array, value) -> array[0] = value;
        SerializableConsumer<Integer> bound = FuncUtils.bind(store, holder);
        bound.accept(12);
        assertEquals(12, holder[0]);

        var first = Numbers.getId();
        var second = Numbers.getId();
        assertEquals(first + 1, second);
    }

    @Test
    void serverLifecycleValidationAndUnknownClientPathsAreCovered() throws Exception {
        assertThrows(IllegalArgumentException.class, () -> new ThreadPoolServer(-1));
        assertThrows(IllegalArgumentException.class, () -> new ThreadPoolServer(65_536));

        try (var executor = Executors.newSingleThreadExecutor()) {
            assertThrows(NullPointerException.class,
                    () -> new ThreadPoolServer(0, null, SocketAccessor.Options.defaults()));
            assertThrows(NullPointerException.class,
                    () -> new ThreadPoolServer(0, executor, null));

            var server = new ThreadPoolServer(0, executor, SocketAccessor.Options.defaults());
            assertThrows(IllegalStateException.class, server::port);
            assertEquals(0, server.clientCount());
            assertTrue(server.clientIds().isEmpty());
            assertThrows(IllegalArgumentException.class, () -> server.callOnClient(99, () -> 1));
            assertThrows(IllegalArgumentException.class,
                    () -> server.executeOnClient(99, ignored -> {}));
            assertThrows(NullPointerException.class,
                    () -> server.callOnClient(99, () -> 1, (Duration) null));
            assertThrows(NullPointerException.class,
                    () -> server.callOnClient(99, () -> 1, (SerializableConsumer<Integer>) null));

            assertSame(server, server.start());
            assertSame(server, server.start());
            assertTrue(server.port() > 0);
            server.close();
            server.close();
        }

        var closed = new ThreadPoolServer(0);
        closed.close();
        assertThrows(IllegalStateException.class, closed::start);
    }

    @Test
    void loopbackWithPlatformTransportCoversCallbacksCommandsAndValidation() throws Exception {
        var options = SocketAccessor.Options.defaults()
                .withWriterBatchSize(1)
                .withVirtualReader(false)
                .withVirtualWriter(false)
                .withInputFilter(info -> ObjectInputFilter.Status.UNDECIDED);

        try (var serverExecutor = Executors.newFixedThreadPool(2);
             var clientExecutor = Executors.newFixedThreadPool(2);
             var server = new ThreadPoolServer(0, serverExecutor, options).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port(), clientExecutor, options)) {
            assertThrows(NullPointerException.class, () -> client.awaitClientId(null));

            var id = client.awaitClientId(Duration.ofSeconds(5));
            assertTrue(id > 0);
            assertEquals(1, server.clientCount());
            assertTrue(server.clientIds().contains(id));

            assertThrows(NullPointerException.class, () -> client.callOnServer(null));
            assertThrows(NullPointerException.class,
                    () -> client.callOnServer(() -> 1, (Duration) null));
            assertThrows(NullPointerException.class,
                    () -> client.callOnServer(() -> 1, (SerializableConsumer<Integer>) null));
            assertThrows(NullPointerException.class, () -> client.executeOnServer(null));

            assertEquals(42, client.callOnServer(() -> 42, Duration.ofSeconds(5)));
            assertEquals(77, server.callOnClient(id, () -> 77, Duration.ofSeconds(5)));

            var clientCallback = client.callOnServer(() -> 5, value -> {
                throw new IOException("client callback");
            });
            var clientCallbackFailure = assertThrows(CompletionException.class, clientCallback::join);
            assertInstanceOf(IOException.class, clientCallbackFailure.getCause());

            var serverCallback = server.callOnClient(id, () -> 6, value -> {
                throw new IOException("server callback");
            });
            var serverCallbackFailure = assertThrows(CompletionException.class, serverCallback::join);
            assertInstanceOf(IOException.class, serverCallbackFailure.getCause());

            client.executeOnServer(context -> {
                if (context.clientCount() != 1) {
                    throw new IllegalStateException("server context");
                }
            }).get(5, TimeUnit.SECONDS);

            server.executeOnClient(id, context -> {
                if (context.statistics().received() == 0) {
                    throw new IllegalStateException("client context");
                }
            }).get(5, TimeUnit.SECONDS);

            var stats = client.statistics();
            assertTrue(stats.sent() > 0);
            assertTrue(stats.received() > 0);
            assertTrue(stats.writerBatches() > 0);
            assertEquals(0, stats.pendingRequests());
        }
    }

    @Test
    void rejectedInvocationExecutorReturnsRemoteFailures() throws Exception {
        var rejected = Executors.newSingleThreadExecutor();
        rejected.shutdown();

        try (var server = new ThreadPoolServer(0, rejected, SocketAccessor.Options.defaults()).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            client.awaitClientId(Duration.ofSeconds(5));

            var call = client.callOnServer(() -> 1);
            var callFailure = assertThrows(CompletionException.class, call::join);
            assertInstanceOf(RemoteExecutionException.class, callFailure.getCause());
            assertEquals(
                    RejectedExecutionException.class.getName(),
                    ((RemoteExecutionException) callFailure.getCause()).remoteType());

            var command = client.executeOnServer(ignored -> {});
            var commandFailure = assertThrows(CompletionException.class, command::join);
            assertInstanceOf(RemoteExecutionException.class, commandFailure.getCause());
        }
    }

    @Test
    void rejectedResponseExecutorFailsPendingRequest() throws Exception {
        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            client.awaitClientId(Duration.ofSeconds(5));

            var connection = connectionOf(client);
            var responseExecutor = responseExecutorOf(connection);
            responseExecutor.close();

            var future = client.callOnServer(() -> 123);
            var failure = assertThrows(CompletionException.class, future::join);
            assertInstanceOf(RejectedExecutionException.class, failure.getCause());
            assertEquals(0, client.statistics().pendingRequests());
        }
    }

    @Test
    void closeBeforeRegistrationFailsClientIdFuture() throws Exception {
        try (var serverSocket = new ServerSocket(0)) {
            var handshake = new CountDownLatch(1);
            var peer = Thread.ofVirtual().start(() -> {
                try (var socket = serverSocket.accept();
                     var out = new ObjectOutputStream(new BufferedOutputStream(socket.getOutputStream()))) {
                    out.flush();
                    try (var in = new ObjectInputStream(new BufferedInputStream(socket.getInputStream()))) {
                        handshake.countDown();
                        try {
                            in.readObject();
                        } catch (EOFException ignored) {
                        }
                    }
                } catch (Exception ignored) {
                    handshake.countDown();
                }
            });

            var client = new ThreadPoolClient("127.0.0.1", serverSocket.getLocalPort());
            assertTrue(handshake.await(5, TimeUnit.SECONDS));
            assertFalse(client.clientId().isDone());

            client.close();
            peer.join();

            assertTrue(client.clientId().isCompletedExceptionally());
            assertThrows(CompletionException.class, client.clientId()::join);
        }
    }

    @Test
    void malformedWireObjectClosesTransportAndExposesTermination() throws Exception {
        try (var serverSocket = new ServerSocket(0)) {
            var sent = new CountDownLatch(1);
            var peer = Thread.ofVirtual().start(() -> {
                try (var socket = serverSocket.accept();
                     var out = new ObjectOutputStream(new BufferedOutputStream(socket.getOutputStream()))) {
                    out.flush();
                    try (var in = new ObjectInputStream(new BufferedInputStream(socket.getInputStream()))) {
                        out.writeObject("not a wire message");
                        out.flush();
                        sent.countDown();
                        try {
                            in.readObject();
                        } catch (Exception ignored) {
                        }
                    }
                } catch (Exception ignored) {
                    sent.countDown();
                }
            });

            var client = new ThreadPoolClient("127.0.0.1", serverSocket.getLocalPort());
            var connection = connectionOf(client);
            try {
                assertTrue(sent.await(5, TimeUnit.SECONDS));
                var deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
                while (!connection.isClosed() && System.nanoTime() < deadline) {
                    Thread.sleep(5);
                }
                assertTrue(connection.isClosed());
                assertTrue(connection.awaitTermination(5, TimeUnit.SECONDS));
                assertThrows(RejectedExecutionException.class, () -> client.callOnServer(() -> 1));
            } finally {
                client.close();
            }
            peer.join();
        }
    }

    @Test
    void clientConnectionFailureAndCloseAreReported() throws Exception {
        var server = new ThreadPoolServer(0).start();
        var port = server.port();
        server.close();
        server.close();

        assertThrows(IOException.class, () -> new ThreadPoolClient("127.0.0.1", port));
        assertThrows(NullPointerException.class, () -> new ThreadPoolClient(null, port));
    }

    private static SocketAccessor<?> connectionOf(ThreadPoolClient client) throws Exception {
        var field = ThreadPoolClient.class.getDeclaredField("connection");
        field.setAccessible(true);
        return (SocketAccessor<?>) field.get(client);
    }

    private static ExecutorService responseExecutorOf(SocketAccessor<?> connection) throws Exception {
        var field = SocketAccessor.class.getDeclaredField("responseExecutor");
        field.setAccessible(true);
        return (ExecutorService) field.get(connection);
    }
}

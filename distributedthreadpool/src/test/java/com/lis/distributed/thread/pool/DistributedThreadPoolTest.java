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
import java.io.ObjectInputFilter;
import java.time.Duration;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
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
    void clientConnectionFailureAndCloseAreReported() throws Exception {
        var server = new ThreadPoolServer(0).start();
        var port = server.port();
        server.close();
        server.close();

        assertThrows(IOException.class, () -> new ThreadPoolClient("127.0.0.1", port));
        assertThrows(NullPointerException.class, () -> new ThreadPoolClient(null, port));
    }
}

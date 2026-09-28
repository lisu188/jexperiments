package com.lis.distributed.thread.pool;

import com.lis.distributed.thread.pool.client.ThreadPoolClient;
import com.lis.distributed.thread.pool.func.FuncUtils;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;
import org.junit.jupiter.api.Test;

import java.io.ObjectInputFilter;
import java.time.Duration;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class DistributedThreadPoolCoverageTest {
    @Test
    void repositoryCoversCompletionFailureCloseAndUnknownIds() {
        var repository = new DataRepository();
        var first = repository.<String>register();
        var second = repository.<Integer>register();

        assertEquals(0, first.id());
        assertEquals(1, second.id());
        assertEquals(2, repository.pendingCount());

        repository.complete(999, "ignored");
        repository.fail(998, new IllegalStateException("ignored"));
        assertEquals(2, repository.pendingCount());

        repository.complete(first.id(), "done");
        assertEquals("done", first.future().join());

        repository.fail(second.id(), new IllegalArgumentException("boom"));
        var failure = assertThrows(CompletionException.class, second.future()::join);
        assertInstanceOf(IllegalArgumentException.class, failure.getCause());
        assertEquals(0, repository.pendingCount());

        assertThrows(NullPointerException.class, () -> repository.fail(1, null));
        assertThrows(NullPointerException.class, () -> repository.failAll(null));

        var third = repository.<Long>register();
        repository.failAll(new IllegalStateException("all"));
        assertTrue(third.future().isCompletedExceptionally());

        repository.close();
        repository.close();
        assertThrows(RejectedExecutionException.class, repository::register);
    }

    @Test
    void wireMessagesValidatePayloadAndPreserveRemoteFailure() {
        assertThrows(IllegalArgumentException.class,
                () -> new WireMessage.Invocation(-1, () -> 1));
        assertThrows(NullPointerException.class,
                () -> new WireMessage.Invocation(0, null));
        assertThrows(IllegalArgumentException.class,
                () -> new WireMessage.Command(-1, context -> {}));
        assertThrows(NullPointerException.class,
                () -> new WireMessage.Command(0, null));
        assertThrows(IllegalArgumentException.class,
                () -> new WireMessage.Response(-1, null, null));
        assertThrows(IllegalArgumentException.class,
                () -> new WireMessage.Response(0, new Object(), null));
        assertThrows(IllegalArgumentException.class,
                () -> new WireMessage.Registration(0));

        var success = WireMessage.Response.success(7, "ok");
        assertEquals(7, success.requestId());
        assertEquals("ok", success.value());
        assertNull(success.failure());

        var source = new IllegalStateException("remote boom");
        var failed = WireMessage.Response.failure(8, source);
        assertNotNull(failed.failure());
        var remote = failed.failure().toException();
        assertEquals(IllegalStateException.class.getName(), remote.remoteType());
        assertTrue(remote.getMessage().contains("remote boom"));
        assertTrue(remote.remoteStackTrace().contains("IllegalStateException"));

        assertThrows(NullPointerException.class,
                () -> new WireMessage.RemoteFailure(null, "x", "stack"));
        assertThrows(NullPointerException.class,
                () -> new WireMessage.RemoteFailure("type", "x", null));
    }

    @Test
    void transportOptionsAndFunctionBindersCoverAllVariants() throws Exception {
        var filter = (ObjectInputFilter) info -> ObjectInputFilter.Status.UNDECIDED;
        var defaults = SocketAccessor.Options.defaults();
        var configured = defaults
                .withIoBufferBytes(1024)
                .withWriterBatchSize(1)
                .withTcpNoDelay(false)
                .withVirtualReader(false)
                .withVirtualWriter(false)
                .withInputFilter(filter);

        assertEquals(1024, configured.ioBufferBytes());
        assertEquals(1, configured.writerBatchSize());
        assertFalse(configured.tcpNoDelay());
        assertFalse(configured.virtualReader());
        assertFalse(configured.virtualWriter());
        assertSame(filter, configured.inputFilter());

        assertThrows(IllegalArgumentException.class,
                () -> new SocketAccessor.Options(1023, 1, true, true, true, null));
        assertThrows(IllegalArgumentException.class,
                () -> new SocketAccessor.Options(1024, 0, true, true, true, null));

        assertEquals(6, FuncUtils.bind((Integer value) -> value * 2, 3).get());
        assertEquals(7, FuncUtils.bind((Integer left, Integer right) -> left + right, 3).apply(4));
        var captured = new int[1];
        FuncUtils.bind((com.lis.distributed.thread.pool.func.SerializableBiConsumer<int[], Integer>)
                ((target, value) -> target[0] = value), captured).accept(9);
        assertEquals(9, captured[0]);

        long first = Numbers.getId();
        assertEquals(first + 1, Numbers.getId());
    }

    @Test
    void configuredPlatformTransportExercisesCallbacksCommandsAndStatistics() throws Exception {
        var executor = Executors.newFixedThreadPool(4);
        var options = SocketAccessor.Options.defaults()
                .withIoBufferBytes(1024)
                .withWriterBatchSize(1)
                .withTcpNoDelay(false)
                .withVirtualReader(false)
                .withVirtualWriter(false)
                .withInputFilter(info -> ObjectInputFilter.Status.UNDECIDED);

        try (executor;
             var server = new ThreadPoolServer(0, executor, options).start()) {
            assertSame(server, server.start());
            assertTrue(server.port() > 0);
            assertEquals(0, server.clientCount());
            assertTrue(server.clientIds().isEmpty());

            try (var client = new ThreadPoolClient("127.0.0.1", server.port(), executor, options)) {
                assertThrows(NullPointerException.class, () -> client.awaitClientId(null));
                int id = client.awaitClientId(Duration.ofSeconds(5));
                assertEquals(1, server.clientCount());
                assertTrue(server.clientIds().contains(id));

                assertEquals(42, client.callOnServer(() -> 42, Duration.ofSeconds(5)));
                assertEquals(5, client.callOnServer(() -> 5, value -> assertEquals(5, value))
                        .get(5, TimeUnit.SECONDS));

                var clientCallbackFailure = client.callOnServer(() -> 1, value -> {
                    throw new IllegalStateException("client callback");
                });
                assertInstanceOf(IllegalStateException.class,
                        assertThrows(CompletionException.class, clientCallbackFailure::join).getCause());

                assertEquals(77, server.callOnClient(id, () -> 77, Duration.ofSeconds(5)));
                assertEquals(8, server.callOnClient(id, () -> 8, value -> assertEquals(8, value))
                        .get(5, TimeUnit.SECONDS));

                var serverCallbackFailure = server.callOnClient(id, () -> 2, value -> {
                    throw new IllegalStateException("server callback");
                });
                assertInstanceOf(IllegalStateException.class,
                        assertThrows(CompletionException.class, serverCallbackFailure::join).getCause());

                client.executeOnServer(context -> {
                    if (context.clientCount() != 1) throw new IllegalStateException();
                }).get(5, TimeUnit.SECONDS);
                server.executeOnClient(id, context -> {
                    if (context.statistics().received() == 0) throw new IllegalStateException();
                }).get(5, TimeUnit.SECONDS);

                assertTrue(client.statistics().sent() > 0);
                assertTrue(client.statistics().received() > 0);
                assertThrows(IllegalArgumentException.class, () -> server.callOnClient(999, () -> 1));
                assertThrows(NullPointerException.class,
                        () -> server.callOnClient(id, () -> 1, (Duration) null));
                assertThrows(NullPointerException.class,
                        () -> client.callOnServer(() -> 1, (Duration) null));
                assertThrows(NullPointerException.class,
                        () -> client.callOnServer(() -> 1, (com.lis.distributed.thread.pool.func.SerializableConsumer<Integer>) null));
                assertThrows(NullPointerException.class,
                        () -> server.callOnClient(id, () -> 1, (com.lis.distributed.thread.pool.func.SerializableConsumer<Integer>) null));
            }

            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (server.clientCount() != 0 && System.nanoTime() < deadline) {
                Thread.onSpinWait();
            }
            assertEquals(0, server.clientCount());
        }
    }

    @Test
    void serverAndClientValidateLifecycleAndRejectedExecutors() throws Exception {
        assertThrows(IllegalArgumentException.class, () -> new ThreadPoolServer(-1));
        assertThrows(IllegalArgumentException.class, () -> new ThreadPoolServer(65_536));
        assertThrows(NullPointerException.class,
                () -> new ThreadPoolServer(0, null, SocketAccessor.Options.defaults()));
        try (var executor = Executors.newSingleThreadExecutor()) {
            assertThrows(NullPointerException.class,
                    () -> new ThreadPoolServer(0, executor, null));
        }

        var unopened = new ThreadPoolServer(0);
        assertThrows(IllegalStateException.class, unopened::port);
        unopened.close();
        unopened.close();
        assertThrows(IllegalStateException.class, unopened::start);

        var rejectedServerExecutor = Executors.newSingleThreadExecutor();
        rejectedServerExecutor.shutdown();
        try (var server = new ThreadPoolServer(
                0, rejectedServerExecutor, SocketAccessor.Options.defaults()).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            client.awaitClientId(Duration.ofSeconds(5));
            var call = client.callOnServer(() -> 1);
            var callFailure = assertThrows(CompletionException.class, call::join);
            assertInstanceOf(RemoteExecutionException.class, callFailure.getCause());

            var command = client.executeOnServer(context -> {});
            var commandFailure = assertThrows(CompletionException.class, command::join);
            assertInstanceOf(RemoteExecutionException.class, commandFailure.getCause());
        }

        var rejectedClientExecutor = Executors.newSingleThreadExecutor();
        rejectedClientExecutor.shutdown();
        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient(
                     "127.0.0.1", server.port(), rejectedClientExecutor, SocketAccessor.Options.defaults())) {
            int id = client.awaitClientId(Duration.ofSeconds(5));
            var call = server.callOnClient(id, () -> 1);
            var failure = assertThrows(CompletionException.class, call::join);
            assertInstanceOf(RemoteExecutionException.class, failure.getCause());
        }

        assertThrows(NullPointerException.class,
                () -> new ThreadPoolClient(null, 1, Executors.newSingleThreadExecutor(), SocketAccessor.Options.defaults()));
    }
}

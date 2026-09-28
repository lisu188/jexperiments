package com.lis.distributed.thread.pool;

import com.lis.distributed.thread.pool.client.ThreadPoolClient;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;

import java.time.Duration;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public final class DistributedThreadPoolVerification {
    public static void main(String[] args) throws Exception {
        verifyRoundTrip();
        verifyConcurrentPipeline();
        verifyRemoteFailure();
        verifyNonSerializableResult();
        verifyServerToClient();
        verifyCommandContext();
        verifyCallbackIsolation();
        verifyMultipleClients();
        verifyCloseFailsPendingRequest();
        System.out.println("DistributedThreadPool verification passed");
    }

    private static void verifyRoundTrip() throws Exception {
        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            require(client.awaitClientId(Duration.ofSeconds(5)) > 0, "client registration");
            require(client.callOnServer(() -> 42, Duration.ofSeconds(5)) == 42, "round trip");
        }
    }

    private static void verifyConcurrentPipeline() throws Exception {
        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            client.awaitClientId(Duration.ofSeconds(5));
            var count = 500;
            var futures = new ArrayList<CompletableFuture<Integer>>(count);
            for (int i = 0; i < count; i++) {
                var value = i;
                futures.add(client.callOnServer(() -> value));
            }
            for (int i = 0; i < count; i++) {
                require(futures.get(i).get(10, TimeUnit.SECONDS) == i, "pipeline result " + i);
            }
            var stats = client.statistics();
            require(stats.pendingRequests() == 0, "all pipelined requests resolved");
            require(stats.sent() >= count, "request writes");
            require(stats.received() >= count, "response reads");
        }
    }

    private static void verifyRemoteFailure() throws Exception {
        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            client.awaitClientId(Duration.ofSeconds(5));
            var future = client.callOnServer(() -> {
                throw new IllegalStateException("remote boom");
            });
            try {
                future.join();
                throw new AssertionError("remote exception expected");
            } catch (CompletionException failure) {
                require(failure.getCause() instanceof RemoteExecutionException, "remote exception type");
                var remote = (RemoteExecutionException) failure.getCause();
                require(remote.remoteType().equals(IllegalStateException.class.getName()), "remote class name");
                require(remote.getMessage().contains("remote boom"), "remote message");
            }
        }
    }

    private static void verifyNonSerializableResult() throws Exception {
        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            client.awaitClientId(Duration.ofSeconds(5));
            var future = client.callOnServer(Object::new);
            try {
                future.join();
                throw new AssertionError("non-serializable result must fail");
            } catch (CompletionException failure) {
                require(failure.getCause() instanceof RemoteExecutionException, "non-serializable remote failure");
            }
        }
    }

    private static void verifyServerToClient() throws Exception {
        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            var id = client.awaitClientId(Duration.ofSeconds(5));
            var value = server.callOnClient(id, () -> 77, Duration.ofSeconds(5));
            require(value == 77, "server-to-client request");
        }
    }

    private static void verifyCommandContext() throws Exception {
        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            var id = client.awaitClientId(Duration.ofSeconds(5));
            client.executeOnServer(context -> {
                if (context.clientCount() != 1) {
                    throw new IllegalStateException("server context not supplied");
                }
            }).get(5, TimeUnit.SECONDS);

            server.executeOnClient(id, context -> {
                if (context.statistics().received() == 0) {
                    throw new IllegalStateException("client context not supplied");
                }
            }).get(5, TimeUnit.SECONDS);
        }
    }

    private static void verifyCallbackIsolation() throws Exception {
        try (var server = new ThreadPoolServer(0).start();
             var client = new ThreadPoolClient("127.0.0.1", server.port())) {
            client.awaitClientId(Duration.ofSeconds(5));
            var callbackEntered = new CountDownLatch(1);
            var releaseCallback = new CountDownLatch(1);

            var first = client.callOnServer(() -> 1);
            first.thenRun(() -> {
                callbackEntered.countDown();
                await(releaseCallback);
            });
            callbackEntered.await();

            var second = client.callOnServer(() -> 2);
            require(second.get(5, TimeUnit.SECONDS) == 2, "blocked callback must not block reader");
            releaseCallback.countDown();
        }
    }

    private static void verifyMultipleClients() throws Exception {
        try (var server = new ThreadPoolServer(0).start();
             var first = new ThreadPoolClient("127.0.0.1", server.port());
             var second = new ThreadPoolClient("127.0.0.1", server.port())) {
            var firstId = first.awaitClientId(Duration.ofSeconds(5));
            var secondId = second.awaitClientId(Duration.ofSeconds(5));
            require(firstId != secondId, "unique client ids");
            require(server.clientCount() == 2, "two registered clients");
            require(server.callOnClient(firstId, () -> 1, Duration.ofSeconds(5)) == 1, "first client route");
            require(server.callOnClient(secondId, () -> 2, Duration.ofSeconds(5)) == 2, "second client route");
        }
    }

    private static void verifyCloseFailsPendingRequest() throws Exception {
        var server = new ThreadPoolServer(0).start();
        var client = new ThreadPoolClient("127.0.0.1", server.port());
        try {
            client.awaitClientId(Duration.ofSeconds(5));
            var future = client.callOnServer(() -> {
                Thread.sleep(500);
                return 1;
            });
            client.close();
            require(future.isCompletedExceptionally(), "pending request failed on close");
        } finally {
            client.close();
            server.close();
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException interruption) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(interruption);
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
}

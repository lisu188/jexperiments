package com.lis.distributed.thread.pool.client;

import com.lis.distributed.thread.pool.SocketAccessor;
import com.lis.distributed.thread.pool.TaskMessage;
import com.lis.distributed.thread.pool.func.SerializableConsumer;
import com.lis.distributed.thread.pool.func.SerializableSupplier;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;

import java.io.IOException;
import java.net.Socket;
import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public final class ThreadPoolClient implements AutoCloseable {
    private final ExecutorService invocationExecutor;
    private final boolean ownsInvocationExecutor;
    private final ClientConnectionProcessor connection;
    private final CompletableFuture<Integer> clientId = new CompletableFuture<>();
    private final AtomicBoolean closed = new AtomicBoolean();

    public ThreadPoolClient(String host, int port) throws IOException {
        this(host, port, Executors.newVirtualThreadPerTaskExecutor(), true, SocketAccessor.Options.defaults());
    }

    public ThreadPoolClient(
            String host,
            int port,
            ExecutorService invocationExecutor,
            SocketAccessor.Options options) throws IOException {
        this(host, port, invocationExecutor, false, options);
    }

    private ThreadPoolClient(
            String host,
            int port,
            ExecutorService invocationExecutor,
            boolean ownsInvocationExecutor,
            SocketAccessor.Options options) throws IOException {
        Objects.requireNonNull(host, "host");
        this.invocationExecutor = Objects.requireNonNull(invocationExecutor, "invocationExecutor");
        this.ownsInvocationExecutor = ownsInvocationExecutor;
        connection = new ClientConnectionProcessor(
                this,
                new Socket(host, port),
                invocationExecutor,
                Objects.requireNonNull(options, "options"));
    }

    public CompletableFuture<Integer> clientId() {
        return clientId;
    }

    public int awaitClientId(Duration timeout) throws Exception {
        Objects.requireNonNull(timeout, "timeout");
        return clientId.get(timeout.toNanos(), TimeUnit.NANOSECONDS);
    }

    public <T> CompletableFuture<T> callOnServer(SerializableSupplier<T> target) {
        return connection.call(target);
    }

    public <T> T callOnServer(SerializableSupplier<T> target, Duration timeout) throws Exception {
        Objects.requireNonNull(timeout, "timeout");
        return callOnServer(target).get(timeout.toNanos(), TimeUnit.NANOSECONDS);
    }

    public <T> CompletableFuture<T> callOnServer(
            SerializableSupplier<T> target,
            SerializableConsumer<T> callback) {
        Objects.requireNonNull(callback, "callback");
        return callOnServer(target).thenApply(value -> {
            try {
                callback.accept(value);
                return value;
            } catch (Exception failure) {
                throw new java.util.concurrent.CompletionException(failure);
            }
        });
    }

    public CompletableFuture<Void> executeOnServer(TaskMessage<ThreadPoolServer> task) {
        return connection.execute(task);
    }

    public SocketAccessor.Statistics statistics() {
        return connection.statistics();
    }

    void setId(int id) {
        clientId.complete(id);
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        connection.close();
        if (!clientId.isDone()) {
            clientId.completeExceptionally(new IOException("client closed before registration"));
        }
        if (ownsInvocationExecutor) {
            invocationExecutor.close();
        }
    }

    public static void main(String[] args) throws Exception {
        try (var client = new ThreadPoolClient("127.0.0.1", 55_555)) {
            System.out.println("client id = " + client.awaitClientId(Duration.ofSeconds(5)));
            for (int i = 0; i < 10; i++) {
                System.out.println("remote result = " + client.callOnServer(() -> 42, Duration.ofSeconds(5)));
            }
        }
    }
}

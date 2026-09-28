package com.lis.distributed.thread.pool.server;

import com.lis.distributed.thread.pool.SocketAccessor;
import com.lis.distributed.thread.pool.TaskMessage;
import com.lis.distributed.thread.pool.client.ThreadPoolClient;
import com.lis.distributed.thread.pool.func.SerializableConsumer;
import com.lis.distributed.thread.pool.func.SerializableSupplier;

import java.io.IOException;
import java.net.ServerSocket;
import java.time.Duration;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public final class ThreadPoolServer implements AutoCloseable {
    private final int configuredPort;
    private final ExecutorService invocationExecutor;
    private final boolean ownsInvocationExecutor;
    private final SocketAccessor.Options transportOptions;
    private final ConcurrentHashMap<Integer, ServerConnectionThread> clients = new ConcurrentHashMap<>();
    private final AtomicInteger nextClientId = new AtomicInteger();
    private final AtomicBoolean started = new AtomicBoolean();
    private final AtomicBoolean closed = new AtomicBoolean();

    private volatile ServerSocket serverSocket;
    private volatile Thread acceptThread;

    public ThreadPoolServer(int port) {
        this(port, Executors.newVirtualThreadPerTaskExecutor(), true, SocketAccessor.Options.defaults());
    }

    public ThreadPoolServer(
            int port,
            ExecutorService invocationExecutor,
            SocketAccessor.Options transportOptions) {
        this(port, invocationExecutor, false, transportOptions);
    }

    private ThreadPoolServer(
            int port,
            ExecutorService invocationExecutor,
            boolean ownsInvocationExecutor,
            SocketAccessor.Options transportOptions) {
        if (port < 0 || port > 65_535) {
            throw new IllegalArgumentException("invalid port: " + port);
        }
        configuredPort = port;
        this.invocationExecutor = Objects.requireNonNull(invocationExecutor, "invocationExecutor");
        this.ownsInvocationExecutor = ownsInvocationExecutor;
        this.transportOptions = Objects.requireNonNull(transportOptions, "transportOptions");
    }

    public ThreadPoolServer start() throws IOException {
        if (!started.compareAndSet(false, true)) {
            return this;
        }
        if (closed.get()) {
            throw new IllegalStateException("server is closed");
        }
        serverSocket = new ServerSocket(configuredPort);
        acceptThread = Thread.ofVirtual().name("distributed-accept-", 0).start(this::acceptLoop);
        return this;
    }

    public int port() {
        var socket = serverSocket;
        if (socket == null) {
            throw new IllegalStateException("server is not started");
        }
        return socket.getLocalPort();
    }

    public int clientCount() {
        return clients.size();
    }

    public Set<Integer> clientIds() {
        return Set.copyOf(clients.keySet());
    }

    public <T> CompletableFuture<T> callOnClient(int clientId, SerializableSupplier<T> target) {
        return requireClient(clientId).call(target);
    }

    public <T> T callOnClient(int clientId, SerializableSupplier<T> target, Duration timeout) throws Exception {
        Objects.requireNonNull(timeout, "timeout");
        return callOnClient(clientId, target).get(timeout.toNanos(), TimeUnit.NANOSECONDS);
    }

    public <T> CompletableFuture<T> callOnClient(
            int clientId,
            SerializableSupplier<T> target,
            SerializableConsumer<T> callback) {
        Objects.requireNonNull(callback, "callback");
        return callOnClient(clientId, target).thenApply(value -> {
            try {
                callback.accept(value);
                return value;
            } catch (Exception failure) {
                throw new java.util.concurrent.CompletionException(failure);
            }
        });
    }

    public CompletableFuture<Void> executeOnClient(int clientId, TaskMessage<ThreadPoolClient> task) {
        @SuppressWarnings("unchecked")
        var adapted = (TaskMessage<ThreadPoolServer>) (TaskMessage<?>) task;
        return requireClient(clientId).execute(adapted);
    }

    void registerClient(int id, ServerConnectionThread connection) {
        var previous = clients.putIfAbsent(id, connection);
        if (previous != null) {
            throw new IllegalStateException("duplicate client id: " + id);
        }
    }

    void unregisterClient(int id, ServerConnectionThread connection) {
        clients.remove(id, connection);
    }

    private ServerConnectionThread requireClient(int clientId) {
        var client = clients.get(clientId);
        if (client == null) {
            throw new IllegalArgumentException("Unknown client id: " + clientId);
        }
        return client;
    }

    private void acceptLoop() {
        while (!closed.get()) {
            try {
                var socket = serverSocket.accept();
                var id = nextClientId.incrementAndGet();
                new ServerConnectionThread(this, socket, id, invocationExecutor, transportOptions);
            } catch (IOException failure) {
                if (!closed.get()) {
                    close();
                }
                return;
            }
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        var socket = serverSocket;
        if (socket != null) {
            try {
                socket.close();
            } catch (IOException ignored) {
            }
        }
        clients.values().forEach(ServerConnectionThread::close);
        clients.clear();

        var thread = acceptThread;
        if (thread != null && thread != Thread.currentThread()) {
            var interrupted = false;
            while (thread.isAlive()) {
                try {
                    thread.join();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }

        if (ownsInvocationExecutor) {
            invocationExecutor.close();
        }
    }

    public static void main(String[] args) throws Exception {
        try (var server = new ThreadPoolServer(55_555).start()) {
            System.out.println("distributed thread pool listening on port " + server.port());
            Thread.currentThread().join();
        }
    }
}

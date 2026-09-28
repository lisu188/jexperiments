package com.lis.distributed.thread.pool;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

public final class DataRepository implements AutoCloseable {
    record Pending<T>(long id, CompletableFuture<T> future) {
    }

    private final AtomicLong nextId = new AtomicLong();
    private final ConcurrentHashMap<Long, CompletableFuture<Object>> pending = new ConcurrentHashMap<>();
    private final AtomicBoolean closed = new AtomicBoolean();

    <T> Pending<T> register() {
        if (closed.get()) {
            throw new RejectedExecutionException("repository is closed");
        }

        var id = nextId.getAndIncrement();
        var future = new CompletableFuture<T>();
        @SuppressWarnings("unchecked")
        var stored = (CompletableFuture<Object>) (CompletableFuture<?>) future;
        if (pending.putIfAbsent(id, stored) != null) {
            throw new IllegalStateException("duplicate request id: " + id);
        }

        if (closed.get() && pending.remove(id, stored)) {
            future.completeExceptionally(new RejectedExecutionException("repository is closed"));
        }
        return new Pending<>(id, future);
    }

    void complete(long id, Object value) {
        var future = pending.remove(id);
        if (future != null) {
            future.complete(value);
        }
    }

    void fail(long id, Throwable failure) {
        Objects.requireNonNull(failure, "failure");
        var future = pending.remove(id);
        if (future != null) {
            future.completeExceptionally(failure);
        }
    }

    void failAll(Throwable failure) {
        Objects.requireNonNull(failure, "failure");
        pending.forEach((id, future) -> {
            if (pending.remove(id, future)) {
                future.completeExceptionally(failure);
            }
        });
    }

    int pendingCount() {
        return pending.size();
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            failAll(new RejectedExecutionException("repository is closed"));
        }
    }
}

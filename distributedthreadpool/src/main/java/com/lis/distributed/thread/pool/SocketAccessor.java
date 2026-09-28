package com.lis.distributed.thread.pool;

import com.lis.distributed.thread.pool.func.SerializableSupplier;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.ObjectInputFilter;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.net.SocketException;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedTransferQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.LongAdder;

public class SocketAccessor<C> implements AutoCloseable {
    public record Options(
            int ioBufferBytes,
            int writerBatchSize,
            boolean tcpNoDelay,
            boolean virtualReader,
            boolean virtualWriter,
            ObjectInputFilter inputFilter) {
        public Options {
            if (ioBufferBytes < 1024) {
                throw new IllegalArgumentException("ioBufferBytes must be >= 1024");
            }
            if (writerBatchSize < 1) {
                throw new IllegalArgumentException("writerBatchSize must be >= 1");
            }
        }

        public static Options defaults() {
            return new Options(64 * 1024, 64, true, true, true, null);
        }

        public Options withWriterBatchSize(int value) {
            return new Options(ioBufferBytes, value, tcpNoDelay, virtualReader, virtualWriter, inputFilter);
        }

        public Options withIoBufferBytes(int value) {
            return new Options(value, writerBatchSize, tcpNoDelay, virtualReader, virtualWriter, inputFilter);
        }

        public Options withTcpNoDelay(boolean value) {
            return new Options(ioBufferBytes, writerBatchSize, value, virtualReader, virtualWriter, inputFilter);
        }

        public Options withVirtualReader(boolean value) {
            return new Options(ioBufferBytes, writerBatchSize, tcpNoDelay, value, virtualWriter, inputFilter);
        }

        public Options withVirtualWriter(boolean value) {
            return new Options(ioBufferBytes, writerBatchSize, tcpNoDelay, virtualReader, value, inputFilter);
        }

        public Options withInputFilter(ObjectInputFilter value) {
            return new Options(ioBufferBytes, writerBatchSize, tcpNoDelay, virtualReader, virtualWriter, value);
        }
    }

    public record Statistics(
            long sent,
            long received,
            long writeFailures,
            long pendingWrites,
            long writerBatches,
            int pendingRequests) {
    }

    private record Outbound(WireMessage message, CompletableFuture<Void> written) {
    }

    private final C context;
    private final Socket socket;
    private final ExecutorService invocationExecutor;
    private final ExecutorService responseExecutor;
    private final Options options;
    private final DataRepository repository = new DataRepository();
    private final BlockingQueue<Outbound> writes = new LinkedTransferQueue<>();
    private final ObjectOutputStream out;
    private final ObjectInputStream in;
    private final AtomicBoolean started = new AtomicBoolean();
    private final AtomicBoolean closed = new AtomicBoolean();
    private final AtomicBoolean closeNotified = new AtomicBoolean();
    private final CountDownLatch readerTerminated = new CountDownLatch(1);
    private final CountDownLatch writerTerminated = new CountDownLatch(1);
    private final LongAdder sent = new LongAdder();
    private final LongAdder received = new LongAdder();
    private final LongAdder writeFailures = new LongAdder();
    private final LongAdder pendingWrites = new LongAdder();
    private final LongAdder writerBatches = new LongAdder();

    private volatile Throwable terminalFailure;
    private volatile Thread readerThread;
    private volatile Thread writerThread;

    protected SocketAccessor(C context, Socket socket, ExecutorService invocationExecutor) throws IOException {
        this(context, socket, invocationExecutor, Options.defaults());
    }

    protected SocketAccessor(
            C context,
            Socket socket,
            ExecutorService invocationExecutor,
            Options options) throws IOException {
        this.context = Objects.requireNonNull(context, "context");
        this.socket = Objects.requireNonNull(socket, "socket");
        this.invocationExecutor = Objects.requireNonNull(invocationExecutor, "invocationExecutor");
        this.options = Objects.requireNonNull(options, "options");
        responseExecutor = Executors.newVirtualThreadPerTaskExecutor();

        socket.setTcpNoDelay(options.tcpNoDelay());
        out = new ObjectOutputStream(new BufferedOutputStream(socket.getOutputStream(), options.ioBufferBytes()));
        out.flush();
        in = new ObjectInputStream(new BufferedInputStream(socket.getInputStream(), options.ioBufferBytes()));
        if (options.inputFilter() != null) {
            in.setObjectInputFilter(options.inputFilter());
        }
    }

    protected final C context() {
        return context;
    }

    protected final void startTransport() {
        if (!started.compareAndSet(false, true)) {
            return;
        }
        readerThread = newThread(options.virtualReader(), "distributed-reader", this::readerLoop);
        writerThread = newThread(options.virtualWriter(), "distributed-writer", this::writerLoop);
        readerThread.start();
        writerThread.start();
    }

    public final <T> CompletableFuture<T> call(SerializableSupplier<T> task) {
        Objects.requireNonNull(task, "task");
        var pending = repository.<T>register();
        var message = new WireMessage.Invocation(pending.id(), task);
        send(message).whenComplete((ignored, failure) -> {
            if (failure != null) {
                repository.fail(pending.id(), unwrapCompletionFailure(failure));
            }
        });
        return pending.future();
    }

    public final CompletableFuture<Void> execute(TaskMessage<?> task) {
        Objects.requireNonNull(task, "task");
        var pending = repository.<Void>register();
        var message = new WireMessage.Command(pending.id(), task);
        send(message).whenComplete((ignored, failure) -> {
            if (failure != null) {
                repository.fail(pending.id(), unwrapCompletionFailure(failure));
            }
        });
        return pending.future();
    }

    protected final CompletableFuture<Void> sendRegistration(int clientId) {
        return send(new WireMessage.Registration(clientId));
    }

    public final Statistics statistics() {
        return new Statistics(
                sent.sum(),
                received.sum(),
                writeFailures.sum(),
                pendingWrites.sum(),
                writerBatches.sum(),
                repository.pendingCount());
    }

    public final boolean isClosed() {
        return closed.get();
    }

    public final boolean awaitTermination(long timeout, TimeUnit unit) throws InterruptedException {
        Objects.requireNonNull(unit, "unit");
        var deadline = System.nanoTime() + unit.toNanos(timeout);
        if (!await(readerTerminated, deadline)) {
            return false;
        }
        return await(writerTerminated, deadline);
    }

    protected void onRegistration(int clientId) {
    }

    protected void onClosed(Throwable failure) {
    }

    private CompletableFuture<Void> send(WireMessage message) {
        Objects.requireNonNull(message, "message");
        if (closed.get()) {
            return CompletableFuture.failedFuture(connectionClosedException());
        }

        var written = new CompletableFuture<Void>();
        var outbound = new Outbound(message, written);
        pendingWrites.increment();
        writes.offer(outbound);

        if (closed.get() && writes.remove(outbound)) {
            pendingWrites.decrement();
            written.completeExceptionally(connectionClosedException());
        } else {
            DistributedThreadPoolJfr.send(message.getClass().getSimpleName(), requestId(message));
        }
        return written;
    }

    private void readerLoop() {
        try {
            while (!closed.get()) {
                var value = in.readObject();
                if (!(value instanceof WireMessage message)) {
                    throw new IOException("Unsupported wire object: " + value.getClass().getName());
                }
                received.increment();
                DistributedThreadPoolJfr.receive(message.getClass().getSimpleName(), requestId(message));
                dispatch(message);
            }
        } catch (EOFException | SocketException expectedOnClose) {
            if (!closed.get()) {
                failConnection(expectedOnClose);
            }
        } catch (Throwable failure) {
            failConnection(failure);
        } finally {
            readerTerminated.countDown();
        }
    }

    private void writerLoop() {
        var batch = new Outbound[options.writerBatchSize()];
        try {
            while (!closed.get() || !writes.isEmpty()) {
                Outbound first;
                try {
                    first = writes.poll(100, TimeUnit.MILLISECONDS);
                } catch (InterruptedException interruption) {
                    if (closed.get()) {
                        continue;
                    }
                    Thread.currentThread().interrupt();
                    throw interruption;
                }
                if (first == null) {
                    continue;
                }

                var count = 0;
                batch[count++] = first;
                while (count < batch.length) {
                    var additional = writes.poll();
                    if (additional == null) {
                        break;
                    }
                    batch[count++] = additional;
                }

                for (int i = 0; i < count; i++) {
                    out.writeObject(batch[i].message());
                }
                out.reset();
                out.flush();
                sent.add(count);
                writerBatches.increment();
                DistributedThreadPoolJfr.writeBatch(count);

                for (int i = 0; i < count; i++) {
                    var outbound = batch[i];
                    batch[i] = null;
                    pendingWrites.decrement();
                    outbound.written().complete(null);
                }
            }
        } catch (Throwable failure) {
            writeFailures.increment();
            for (int i = 0; i < batch.length; i++) {
                var outbound = batch[i];
                if (outbound != null) {
                    batch[i] = null;
                    pendingWrites.decrement();
                    outbound.written().completeExceptionally(failure);
                }
            }
            failQueuedWrites(failure);
            failConnection(failure);
        } finally {
            writerTerminated.countDown();
        }
    }

    private void dispatch(WireMessage message) {
        switch (message) {
            case WireMessage.Invocation invocation -> dispatchInvocation(invocation);
            case WireMessage.Command command -> dispatchCommand(command);
            case WireMessage.Response response -> dispatchResponse(response);
            case WireMessage.Registration registration -> onRegistration(registration.clientId());
        }
    }

    private void dispatchInvocation(WireMessage.Invocation invocation) {
        try {
            invocationExecutor.execute(() -> executeInvocation(invocation));
        } catch (RejectedExecutionException rejection) {
            send(WireMessage.Response.failure(invocation.requestId(), rejection));
        }
    }

    private void executeInvocation(WireMessage.Invocation invocation) {
        var event = DistributedThreadPoolJfr.remoteExecution(invocation.requestId());
        var failed = false;
        try {
            var value = invocation.task().get();
            if (value != null && !(value instanceof java.io.Serializable)) {
                throw new IOException("Remote result is not Serializable: " + value.getClass().getName());
            }
            send(WireMessage.Response.success(invocation.requestId(), value));
        } catch (Throwable failure) {
            failed = true;
            send(WireMessage.Response.failure(invocation.requestId(), failure));
        } finally {
            DistributedThreadPoolJfr.commitRemoteExecution(event, failed);
        }
    }

    private void dispatchCommand(WireMessage.Command command) {
        try {
            invocationExecutor.execute(() -> executeCommand(command));
        } catch (RejectedExecutionException rejection) {
            send(WireMessage.Response.failure(command.requestId(), rejection));
        }
    }

    @SuppressWarnings("unchecked")
    private void executeCommand(WireMessage.Command command) {
        var event = DistributedThreadPoolJfr.remoteExecution(command.requestId());
        var failed = false;
        try {
            ((TaskMessage<C>) command.task()).process(context);
            send(WireMessage.Response.success(command.requestId(), null));
        } catch (Throwable failure) {
            failed = true;
            send(WireMessage.Response.failure(command.requestId(), failure));
        } finally {
            DistributedThreadPoolJfr.commitRemoteExecution(event, failed);
        }
    }

    private void dispatchResponse(WireMessage.Response response) {
        try {
            responseExecutor.execute(() -> {
                if (response.failure() == null) {
                    repository.complete(response.requestId(), response.value());
                } else {
                    repository.fail(response.requestId(), response.failure().toException());
                }
            });
        } catch (RejectedExecutionException rejection) {
            repository.fail(response.requestId(), rejection);
        }
    }

    private void failConnection(Throwable failure) {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        terminalFailure = failure;
        repository.failAll(failure);
        failQueuedWrites(failure);
        try {
            socket.close();
        } catch (IOException ignored) {
        }
        var writer = writerThread;
        if (writer != null) {
            writer.interrupt();
        }
        notifyClosed(failure);
    }

    private void failQueuedWrites(Throwable failure) {
        Outbound outbound;
        while ((outbound = writes.poll()) != null) {
            pendingWrites.decrement();
            outbound.written().completeExceptionally(failure);
        }
    }

    @Override
    public final void close() {
        if (closed.compareAndSet(false, true)) {
            terminalFailure = new IOException("connection closed");
            repository.failAll(terminalFailure);
            var writer = writerThread;
            if (writer != null) {
                writer.interrupt();
            }
        }

        awaitUninterruptibly(writerTerminated);
        try {
            socket.close();
        } catch (IOException ignored) {
        }
        awaitUninterruptibly(readerTerminated);

        repository.close();
        responseExecutor.close();
        notifyClosed(terminalFailure);
    }

    private void notifyClosed(Throwable failure) {
        if (closeNotified.compareAndSet(false, true)) {
            onClosed(failure);
        }
    }

    private RuntimeException connectionClosedException() {
        var failure = terminalFailure;
        if (failure == null) {
            return new RejectedExecutionException("connection is closed");
        }
        return new RejectedExecutionException("connection is closed", failure);
    }

    private static long requestId(WireMessage message) {
        return switch (message) {
            case WireMessage.Invocation invocation -> invocation.requestId();
            case WireMessage.Command command -> command.requestId();
            case WireMessage.Response response -> response.requestId();
            case WireMessage.Registration ignored -> -1;
        };
    }

    private static Throwable unwrapCompletionFailure(Throwable failure) {
        return failure instanceof java.util.concurrent.CompletionException completion && completion.getCause() != null
                ? completion.getCause()
                : failure;
    }

    private static boolean await(CountDownLatch latch, long deadline) throws InterruptedException {
        var remaining = deadline - System.nanoTime();
        return remaining > 0 && latch.await(remaining, TimeUnit.NANOSECONDS);
    }

    private static void awaitUninterruptibly(CountDownLatch latch) {
        var interrupted = false;
        while (latch.getCount() != 0) {
            try {
                latch.await();
            } catch (InterruptedException interruption) {
                interrupted = true;
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private static Thread newThread(boolean virtual, String name, Runnable task) {
        ThreadFactory factory = virtual
                ? Thread.ofVirtual().name(name + "-", 0).factory()
                : Thread.ofPlatform().daemon(true).name(name + "-", 0).factory();
        return factory.newThread(task);
    }
}

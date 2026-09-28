package com.lis.threadpool;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.LinkedTransferQueue;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(3)
@State(Scope.Benchmark)
public class OrderedThreadPoolExecutorJmhBenchmark {
    private static final Supplier<Integer> VALUE = () -> 1;

    @Param({"fixed", "virtual"})
    public String workerModel;

    @Param({"linked", "array", "transfer", "synchronous"})
    public String queueModel;

    @Param({"true", "false"})
    public boolean virtualPublisher;

    @Param({"1", "64", "256"})
    public int batchSize;

    private ExecutorService workers;
    private OrderedThreadPoolExecutor<Integer> executor;
    private BlockingQueue<Integer> output;
    private Thread consumer;
    private volatile boolean stopping;
    private final LongAdder consumed = new LongAdder();

    @Setup(Level.Iteration)
    public void setup() {
        workers = "virtual".equals(workerModel)
                ? Executors.newVirtualThreadPerTaskExecutor()
                : Executors.newFixedThreadPool(Math.max(2, Math.min(Runtime.getRuntime().availableProcessors(), 16)));
        output = switch (queueModel) {
            case "linked" -> new LinkedBlockingQueue<>();
            case "array" -> new ArrayBlockingQueue<>(1024);
            case "transfer" -> new LinkedTransferQueue<>();
            case "synchronous" -> new SynchronousQueue<>();
            default -> throw new IllegalArgumentException(queueModel);
        };
        var options = OrderedThreadPoolExecutor.Options.defaults()
                .withVirtualPublisher(virtualPublisher)
                .withPublisherBatchSize(batchSize);
        executor = new OrderedThreadPoolExecutor<>(output, workers, options);
        stopping = false;
        consumer = Thread.ofPlatform().daemon(true).start(() -> {
            try {
                while (!stopping || !output.isEmpty()) {
                    if (output.poll(100, TimeUnit.MILLISECONDS) != null) {
                        consumed.increment();
                    }
                }
            } catch (InterruptedException interruption) {
                Thread.currentThread().interrupt();
            }
        });
    }

    @TearDown(Level.Iteration)
    public void tearDown() throws Exception {
        executor.close();
        stopping = true;
        consumer.join(TimeUnit.SECONDS.toMillis(10));
        workers.close();
    }

    @Benchmark
    public long fireAndForget() {
        return executor.executeOrdered(VALUE);
    }

    @Benchmark
    public CompletableFuture<Integer> future() {
        return executor.process(VALUE);
    }
}

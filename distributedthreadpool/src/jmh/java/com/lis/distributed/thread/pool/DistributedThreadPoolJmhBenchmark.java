package com.lis.distributed.thread.pool;

import com.lis.distributed.thread.pool.client.ThreadPoolClient;
import com.lis.distributed.thread.pool.server.ThreadPoolServer;

import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
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

@BenchmarkMode(Mode.SampleTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(3)
@State(Scope.Thread)
public class DistributedThreadPoolJmhBenchmark {
    @Param({"1", "16", "64"})
    public int batchSize;

    @Param({"true", "false"})
    public boolean tcpNoDelay;

    @Param({"fixed", "virtual"})
    public String workerModel;

    private ExecutorService serverWorkers;
    private ExecutorService clientWorkers;
    private ThreadPoolServer server;
    private ThreadPoolClient client;

    @Setup(Level.Trial)
    public void setup() throws Exception {
        var options = SocketAccessor.Options.defaults()
                .withWriterBatchSize(batchSize)
                .withTcpNoDelay(tcpNoDelay);
        serverWorkers = workers(workerModel);
        clientWorkers = workers(workerModel);
        server = new ThreadPoolServer(0, serverWorkers, options).start();
        client = new ThreadPoolClient("127.0.0.1", server.port(), clientWorkers, options);
        client.awaitClientId(Duration.ofSeconds(5));
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        client.close();
        server.close();
        clientWorkers.close();
        serverWorkers.close();
    }

    @Benchmark
    public int roundTrip() {
        return client.callOnServer(() -> 1).join();
    }

    private static ExecutorService workers(String model) {
        return switch (model) {
            case "fixed" -> Executors.newFixedThreadPool(
                    Math.max(2, Math.min(Runtime.getRuntime().availableProcessors(), 16)));
            case "virtual" -> Executors.newVirtualThreadPerTaskExecutor();
            default -> throw new IllegalArgumentException(model);
        };
    }
}

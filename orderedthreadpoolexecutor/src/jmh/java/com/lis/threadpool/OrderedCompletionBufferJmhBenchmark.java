package com.lis.threadpool;

import java.util.HashMap;
import java.util.SplittableRandom;
import java.util.TreeMap;
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
import org.openjdk.jmh.annotations.Warmup;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(3)
@State(Scope.Thread)
public class OrderedCompletionBufferJmhBenchmark {
    @Param({"1024", "65536"})
    public int size;

    @Param({"ordered", "reverse", "shuffle"})
    public String order;

    private int[] completionOrder;
    private Object[] values;

    @Setup(Level.Trial)
    public void setup() {
        completionOrder = new int[size];
        values = new Object[size];
        for (int i = 0; i < size; i++) {
            completionOrder[i] = i;
            values[i] = new Object();
        }
        if ("reverse".equals(order)) {
            for (int i = 0; i < size / 2; i++) {
                var other = size - i - 1;
                var value = completionOrder[i];
                completionOrder[i] = completionOrder[other];
                completionOrder[other] = value;
            }
        } else if ("shuffle".equals(order)) {
            var random = new SplittableRandom(0x5eedL);
            for (int i = size - 1; i > 0; i--) {
                var other = random.nextInt(i + 1);
                var value = completionOrder[i];
                completionOrder[i] = completionOrder[other];
                completionOrder[other] = value;
            }
        }
    }

    @Benchmark
    public long segmented() {
        var buffer = new OrderedCompletionBuffer<Object>(1024);
        long next = 0;
        long sum = 0;
        for (var sequence : completionOrder) {
            buffer.put(sequence, values[sequence]);
            while (buffer.remove(next) != null) {
                sum += next++;
            }
        }
        return sum;
    }

    @Benchmark
    public long hashMap() {
        var buffer = new HashMap<Long, Object>();
        long next = 0;
        long sum = 0;
        for (var sequence : completionOrder) {
            buffer.put((long) sequence, values[sequence]);
            while (buffer.remove(next) != null) {
                sum += next++;
            }
        }
        return sum;
    }

    @Benchmark
    public long treeMap() {
        var buffer = new TreeMap<Long, Object>();
        long next = 0;
        long sum = 0;
        for (var sequence : completionOrder) {
            buffer.put((long) sequence, values[sequence]);
            while (!buffer.isEmpty() && buffer.firstKey() == next) {
                buffer.remove(next);
                sum += next++;
            }
        }
        return sum;
    }
}

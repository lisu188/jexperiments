package com.lis.threadpool;

import java.util.Objects;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;

public class OrderedThreadPool<T> {
    private final AtomicInteger _counter = new AtomicInteger();
    private final BlockingQueue<T> _queue;
    private final ExecutorService _executor;
    private final SortedMap<Integer, T> _results = new TreeMap<>();
    private int _nextId = 0;
    private final Executor _finalizer = Executors.newSingleThreadExecutor();

    public OrderedThreadPool(BlockingQueue<T> queue, ExecutorService executor) {
        _queue = queue;
        _executor = executor;
    }

    public void process(Supplier<T> task) {
        int id = _counter.getAndIncrement();
        _executor.execute(() -> {
            T result = task.get();
            _finalizer.execute(new Runnable() {
                @Override
                public void run() {
                    try {
                        _results.put(id, result);
                        Integer next = _results.firstKey();
                        while (next == _nextId) {
                            _nextId++;
                            _queue.put(_results.remove(next));
                            next = _results.isEmpty() ? -1 : _results.firstKey();
                        }
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    }
                }
            });
        });
    }

    private final static class TemporaryTask implements Supplier<Integer> {

        private final Integer _i;

        private TemporaryTask(Integer i) {
            _i = i;
        }

        @Override
        public Integer get() {
            try {
                Thread.sleep(ThreadLocalRandom.current().nextInt(100));
                return _i;
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    public static void main(String... args) throws InterruptedException {
        int tasks;
        if (args.length == 1) {
            tasks = Integer.parseInt(args[0]);
        } else {
            tasks = 1000;
        }
        BlockingQueue<Integer> integers = new ArrayBlockingQueue<>(1000);
        OrderedThreadPool<Integer> objectOrderedThreadPool = new OrderedThreadPool<>(integers, Executors.newCachedThreadPool());
        for (int i = 0; i < tasks; i++) {
            objectOrderedThreadPool.process(new TemporaryTask(i));
        }
        for (int i = 0; i < tasks; i++) {
            if (!Objects.equals(i, integers.take())) {
                System.exit(1);
            }
        }
    }
}

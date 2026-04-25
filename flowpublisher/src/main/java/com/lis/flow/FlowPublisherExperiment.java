package com.lis.flow;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Flow;
import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.TimeUnit;

public final class FlowPublisherExperiment {
    private FlowPublisherExperiment() {
        throw new AssertionError();
    }

    public static void main(String[] args) throws InterruptedException {
        List<MetricReading> readings = Arrays.asList(
                new MetricReading("load", 0.41),
                new MetricReading("load", 0.46),
                new MetricReading("load", 0.38),
                new MetricReading("load", 0.52),
                new MetricReading("load", 0.47),
                new MetricReading("load", 0.43)
        );

        ExecutorService executor = Executors.newFixedThreadPool(2);
        CountDownLatch completed = new CountDownLatch(2);

        try {
            try (SubmissionPublisher<MetricReading> publisher = new SubmissionPublisher<>(executor, 2)) {
                publisher.subscribe(new WindowedAverageSubscriber("fast", completed, 3, 0));
                publisher.subscribe(new WindowedAverageSubscriber("slow", completed, 2, 75));

                for (MetricReading reading : readings) {
                    int estimatedLag = publisher.submit(reading);
                    System.out.println("published " + reading + " with max lag " + estimatedLag);
                }
            }

            if (!completed.await(5, TimeUnit.SECONDS)) {
                throw new IllegalStateException("Subscribers did not complete in time");
            }
        } finally {
            executor.shutdown();
            if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        }
    }

    private static final class MetricReading {
        private final String name;
        private final double value;

        private MetricReading(String name, double value) {
            this.name = name;
            this.value = value;
        }

        @Override
        public String toString() {
            return name + "=" + String.format(Locale.ROOT, "%.2f", value);
        }
    }

    private static final class WindowedAverageSubscriber implements Flow.Subscriber<MetricReading> {
        private final String name;
        private final CountDownLatch completed;
        private final int requestWindow;
        private final long processingDelayMillis;
        private int received;
        private int receivedInWindow;
        private double total;
        private Flow.Subscription subscription;

        private WindowedAverageSubscriber(String name,
                                          CountDownLatch completed,
                                          int requestWindow,
                                          long processingDelayMillis) {
            this.name = name;
            this.completed = completed;
            this.requestWindow = requestWindow;
            this.processingDelayMillis = processingDelayMillis;
        }

        @Override
        public void onSubscribe(Flow.Subscription subscription) {
            this.subscription = subscription;
            System.out.println(name + " subscribed and requested " + requestWindow);
            subscription.request(requestWindow);
        }

        @Override
        public void onNext(MetricReading item) {
            if (processingDelayMillis > 0) {
                try {
                    Thread.sleep(processingDelayMillis);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    subscription.cancel();
                    onError(e);
                    return;
                }
            }

            received++;
            receivedInWindow++;
            total += item.value;

            System.out.println(name + " received " + item
                    + " average=" + String.format(Locale.ROOT, "%.2f", total / received));

            if (receivedInWindow == requestWindow) {
                receivedInWindow = 0;
                System.out.println(name + " requested " + requestWindow + " more");
                subscription.request(requestWindow);
            }
        }

        @Override
        public void onError(Throwable throwable) {
            throwable.printStackTrace();
            completed.countDown();
        }

        @Override
        public void onComplete() {
            System.out.println(name + " completed after " + received + " readings");
            completed.countDown();
        }
    }
}

package com.lis.flow;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Flow;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlowPublisherExperimentTest {
    @Test
    void utilityConstructorIsNotInstantiable() throws Exception {
        Constructor<FlowPublisherExperiment> constructor = FlowPublisherExperiment.class.getDeclaredConstructor();
        constructor.setAccessible(true);

        InvocationTargetException failure = assertThrows(InvocationTargetException.class, constructor::newInstance);

        assertTrue(failure.getCause() instanceof AssertionError);
    }

    @Test
    void readingFormattingAndSubscriberWindowingAreCovered() throws Exception {
        Class<?> readingType = Class.forName("com.lis.flow.FlowPublisherExperiment$MetricReading");
        Constructor<?> readingConstructor = readingType.getDeclaredConstructor(String.class, double.class);
        readingConstructor.setAccessible(true);
        Object reading = readingConstructor.newInstance("cpu", 0.5);

        assertEquals("cpu=0.50", reading.toString());

        CountDownLatch completed = new CountDownLatch(1);
        Object subscriber = newSubscriber(completed, 2, 0);
        RecordingSubscription subscription = new RecordingSubscription();

        invoke(subscriber, "onSubscribe", Flow.Subscription.class, subscription);
        invoke(subscriber, "onNext", readingType, reading);
        invoke(subscriber, "onNext", readingType, reading);
        invoke(subscriber, "onComplete");

        assertEquals(4, subscription.requested.get());
        assertTrue(completed.await(1, TimeUnit.SECONDS));
    }

    @Test
    void subscriberErrorAndInterruptPathsCancelAndComplete() throws Exception {
        Class<?> readingType = Class.forName("com.lis.flow.FlowPublisherExperiment$MetricReading");
        Constructor<?> readingConstructor = readingType.getDeclaredConstructor(String.class, double.class);
        readingConstructor.setAccessible(true);
        Object reading = readingConstructor.newInstance("load", 1.0);

        CountDownLatch completed = new CountDownLatch(1);
        Object subscriber = newSubscriber(completed, 1, 1_000);
        RecordingSubscription subscription = new RecordingSubscription();

        invoke(subscriber, "onSubscribe", Flow.Subscription.class, subscription);

        Thread.currentThread().interrupt();
        try {
            invoke(subscriber, "onNext", readingType, reading);
        } finally {
            Thread.interrupted();
        }

        assertTrue(subscription.cancelled.get());
        assertTrue(completed.await(1, TimeUnit.SECONDS));

        CountDownLatch failed = new CountDownLatch(1);
        Object errorSubscriber = newSubscriber(failed, 1, 0);
        invoke(errorSubscriber, "onError", Throwable.class, new IllegalStateException("expected"));
        assertTrue(failed.await(1, TimeUnit.SECONDS));
    }

    private static Object newSubscriber(CountDownLatch completed, int requestWindow, long delayMillis) throws Exception {
        Class<?> subscriberType = Class.forName("com.lis.flow.FlowPublisherExperiment$WindowedAverageSubscriber");
        Constructor<?> constructor = subscriberType.getDeclaredConstructor(
                String.class, CountDownLatch.class, int.class, long.class);
        constructor.setAccessible(true);
        return constructor.newInstance("test", completed, requestWindow, delayMillis);
    }

    private static void invoke(Object target, String methodName, Class<?> parameterType, Object argument)
            throws Exception {
        var method = target.getClass().getDeclaredMethod(methodName, parameterType);
        method.setAccessible(true);
        method.invoke(target, argument);
    }

    private static void invoke(Object target, String methodName) throws Exception {
        var method = target.getClass().getDeclaredMethod(methodName);
        method.setAccessible(true);
        method.invoke(target);
    }

    private static final class RecordingSubscription implements Flow.Subscription {
        private final AtomicLong requested = new AtomicLong();
        private final AtomicBoolean cancelled = new AtomicBoolean();

        @Override
        public void request(long n) {
            requested.addAndGet(n);
        }

        @Override
        public void cancel() {
            cancelled.set(true);
        }
    }
}

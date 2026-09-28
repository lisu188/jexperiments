package com.lis.distributed.thread.pool.func;

public final class FuncUtils {
    private FuncUtils() {
    }

    public static <U, V> SerializableSupplier<V> bind(SerializableFunction<U, V> function, U argument) {
        return () -> function.apply(argument);
    }

    public static <T, U, R> SerializableFunction<U, R> bind(
            SerializableBiFunction<T, U, R> function,
            T argument) {
        return value -> function.apply(argument, value);
    }

    public static <U, V> SerializableConsumer<V> bind(
            SerializableBiConsumer<U, V> consumer,
            U argument) {
        return value -> consumer.accept(argument, value);
    }
}

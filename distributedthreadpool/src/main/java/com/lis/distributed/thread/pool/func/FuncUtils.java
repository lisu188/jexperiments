package com.lis.distributed.thread.pool.func;

public abstract class FuncUtils {
    public static <U, V> SerializableSupplier<V> bind(
            SerializableFunction<U, V> func, U arg) {
        return () -> {
            return func.apply(arg);
        };
    }

    public static <T, U, R> SerializableFunction<U, R> bind(
            SerializableBiFunction<T, U, R> func, T arg) {
        return (x) -> {
            return func.apply(arg, x);
        };
    }

    public static <U, V> SerializableConsumer<V> bind(
            SerializableBiConsumer<U, V> func, U arg) {
        return (x) -> {
            func.accept(arg, x);
        };
    }

}

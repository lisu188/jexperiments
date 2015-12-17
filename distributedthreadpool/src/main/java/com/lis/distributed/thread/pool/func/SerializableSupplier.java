package com.lis.distributed.thread.pool.func;

import java.io.Serializable;

@FunctionalInterface
public interface SerializableSupplier<T> extends Serializable {
    T get() throws Exception;
}

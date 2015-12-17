package com.lis.distributed.thread.pool.func;

import java.io.Serializable;

@FunctionalInterface
public interface SerializableBiConsumer<T1, T2> extends Serializable {
    void accept(T1 t, T2 u) throws Exception;
}

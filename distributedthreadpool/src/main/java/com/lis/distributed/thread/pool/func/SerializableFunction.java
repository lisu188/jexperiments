package com.lis.distributed.thread.pool.func;

import java.io.Serializable;

@FunctionalInterface
public interface SerializableFunction<U, V> extends Serializable {
    V apply(U t) throws Exception;
}

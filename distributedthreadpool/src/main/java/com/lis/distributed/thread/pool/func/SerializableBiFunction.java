package com.lis.distributed.thread.pool.func;

import java.io.Serializable;

@FunctionalInterface
public interface SerializableBiFunction<T, U, R> extends Serializable {
	R apply(T t, U u) throws Exception;
}

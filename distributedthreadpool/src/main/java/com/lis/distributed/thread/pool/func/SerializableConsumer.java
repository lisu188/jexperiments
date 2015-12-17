package com.lis.distributed.thread.pool.func;

import java.io.Serializable;

@FunctionalInterface
public interface SerializableConsumer<T> extends Serializable {
	void accept(T t) throws Exception;
}

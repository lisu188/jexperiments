package com.lis.distributed.thread.pool;

import java.io.Serializable;

@FunctionalInterface
public interface TaskMessage<T> extends Serializable {
	void process(T context) throws Exception;
}

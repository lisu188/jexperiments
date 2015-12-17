package com.lis.distributed.thread.pool;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;

public class DataRepository {
	private ConcurrentHashMap<Integer, CountDownLatch> locks = new ConcurrentHashMap<Integer, CountDownLatch>();

	private ConcurrentHashMap<Integer, Object> values = new ConcurrentHashMap<Integer, Object>();

	public int lock() {
		int id = Numbers.getId();
		locks.put(id, new CountDownLatch(1));
		return id;
	}

	public void setValue(int msgId, Object object) {
		values.put(msgId, object);
		locks.get(msgId).countDown();
	}

	public Object getValue(int msgId) throws Exception {
		locks.get(msgId).await();
		locks.remove(msgId);
		return values.remove(msgId);
	}
}

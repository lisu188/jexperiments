package com.lis.pi.monitor;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.apache.logging.log4j.LogManager;

public class MonitorData implements Serializable {
	public class DataUnit implements Serializable {
		long timeStamp;
		ValueType type;
		float value;

		public DataUnit(long timeStamp, ValueType type, float value) {
			super();
			this.timeStamp = timeStamp;
			this.type = type;
			this.value = value;
		}

		public long getTimeStamp() {
			return timeStamp;
		}

		public ValueType getType() {
			return type;
		}

		public float getValue() {
			return value;
		}

		public void setTimeStamp(long timeStamp) {
			this.timeStamp = timeStamp;
		}

		public void setType(ValueType type) {
			this.type = type;
		}

		public void setValue(float value) {
			this.value = value;
		}
	};

	private static MonitorData instance;

	public static synchronized MonitorData getInstance() {
		if (instance == null) {
			instance = new MonitorData();
		}
		return instance;
	}

	private final Lock lock = new ReentrantLock();

	private final List<DataUnit> polledData = new LinkedList<DataUnit>();

	public List<DataUnit> getPolledData() {
		return polledData;
	}

	public void updateData(ValueType type, float value) {
		lock.lock();
		try {
			polledData
			.add(new DataUnit(System.currentTimeMillis(), type, value));
			if (polledData.size() > 10000) {
				LogManager.getLogger().info("Purging polled data");
				polledData.clear();
			}
		} finally {
			lock.unlock();
		}
	}
}

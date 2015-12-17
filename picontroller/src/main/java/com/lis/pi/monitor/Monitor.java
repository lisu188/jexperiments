package com.lis.pi.monitor;

import com.lis.pi.websocket.InfoEndpoint;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public abstract class Monitor extends Thread {
    private static Set<Monitor> monitors = Collections
            .synchronizedSet(new HashSet<Monitor>());
    private final Lock lock = new ReentrantLock();
    private final ValueType type;
    protected volatile boolean terminated;
    private float value;

    public Monitor(ValueType type) {
        lock.lock();
        try {
            this.setName(getClass().getSimpleName());
            this.type = type;
        } finally {
            lock.unlock();
        }
    }

    public static Set<Monitor> getMonitors() {
        return monitors;
    }

    public static void startMonitors() {
        monitors.add(new LightMonitor());
        monitors.add(new MemMonitor());
        monitors.add(new TempMonitor());
        for (Monitor monitor : monitors) {
            monitor.start();
        }
    }

    public static void stopMonitors() {
        for (Monitor monitor : monitors) {
            monitor.terminate();
        }
    }

    public String getValueString() {
        lock.lock();
        try {
            String json = "{" + "\"type\":\"" + type.toString()
                    + "\",\"value\":" + String.valueOf(value) + "}";
            return json;
        } finally {
            lock.unlock();
        }
    }

    protected void setValue(float value) {
        lock.lock();
        try {
            if (value != this.value) {
                this.value = value;
                valueChanged();
            }
        } finally {
            lock.unlock();
        }
    }

    private void terminate() {
        this.terminated = true;
    }

    private final void valueChanged() {
        lock.lock();
        try {
            MonitorData.getInstance().updateData(type, value);
            InfoEndpoint.broadcast(getValueString());
        } finally {
            lock.unlock();
        }
    }
}

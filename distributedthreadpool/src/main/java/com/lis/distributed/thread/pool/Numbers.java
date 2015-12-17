package com.lis.distributed.thread.pool;

import java.util.concurrent.atomic.AtomicInteger;

public abstract class Numbers {
    private final static AtomicInteger ID_GEN = new AtomicInteger();

    public static int getId() {
        return ID_GEN.incrementAndGet();
    }
}

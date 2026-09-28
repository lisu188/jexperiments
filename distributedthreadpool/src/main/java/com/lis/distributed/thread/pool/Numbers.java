package com.lis.distributed.thread.pool;

import java.util.concurrent.atomic.AtomicLong;

public final class Numbers {
    private static final AtomicLong ID_GEN = new AtomicLong();

    private Numbers() {
    }

    public static long getId() {
        return ID_GEN.incrementAndGet();
    }
}

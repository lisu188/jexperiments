package com.lis.distributed.thread.pool;

import java.io.Serial;

public final class RemoteExecutionException extends RuntimeException {
    @Serial
    private static final long serialVersionUID = 1L;

    private final String remoteType;
    private final String remoteStackTrace;

    RemoteExecutionException(String remoteType, String message, String remoteStackTrace) {
        super(remoteType + (message == null ? "" : ": " + message));
        this.remoteType = remoteType;
        this.remoteStackTrace = remoteStackTrace;
    }

    public String remoteType() {
        return remoteType;
    }

    public String remoteStackTrace() {
        return remoteStackTrace;
    }
}

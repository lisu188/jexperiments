package com.lis.distributed.thread.pool;

import jdk.jfr.Category;
import jdk.jfr.Enabled;
import jdk.jfr.Event;
import jdk.jfr.EventType;
import jdk.jfr.Label;
import jdk.jfr.Name;

final class DistributedThreadPoolJfr {
    private static final EventType SEND = EventType.getEventType(SendEvent.class);
    private static final EventType RECEIVE = EventType.getEventType(ReceiveEvent.class);
    private static final EventType REMOTE_EXECUTION = EventType.getEventType(RemoteExecutionEvent.class);
    private static final EventType WRITE_BATCH = EventType.getEventType(WriteBatchEvent.class);

    private DistributedThreadPoolJfr() {
    }

    static void send(String type, long requestId) {
        if (SEND.isEnabled()) {
            var event = new SendEvent();
            event.type = type;
            event.requestId = requestId;
            event.commit();
        }
    }

    static void receive(String type, long requestId) {
        if (RECEIVE.isEnabled()) {
            var event = new ReceiveEvent();
            event.type = type;
            event.requestId = requestId;
            event.commit();
        }
    }

    static RemoteExecutionEvent remoteExecution(long requestId) {
        if (!REMOTE_EXECUTION.isEnabled()) {
            return null;
        }
        var event = new RemoteExecutionEvent();
        event.requestId = requestId;
        return event;
    }

    static void commitRemoteExecution(RemoteExecutionEvent event, boolean failed) {
        if (event != null) {
            event.failed = failed;
            event.end();
            if (event.shouldCommit()) {
                event.commit();
            }
        }
    }

    static void writeBatch(int messages) {
        if (WRITE_BATCH.isEnabled()) {
            var event = new WriteBatchEvent();
            event.messages = messages;
            event.commit();
        }
    }

    @Name("experiments.DistributedSend")
    @Label("Distributed send")
    @Category({"JExperiments", "DistributedThreadPool"})
    @Enabled(false)
    static final class SendEvent extends Event {
        @Label("Type") String type;
        @Label("Request id") long requestId;
    }

    @Name("experiments.DistributedReceive")
    @Label("Distributed receive")
    @Category({"JExperiments", "DistributedThreadPool"})
    @Enabled(false)
    static final class ReceiveEvent extends Event {
        @Label("Type") String type;
        @Label("Request id") long requestId;
    }

    @Name("experiments.DistributedRemoteExecution")
    @Label("Distributed remote execution")
    @Category({"JExperiments", "DistributedThreadPool"})
    @Enabled(false)
    static final class RemoteExecutionEvent extends Event {
        @Label("Request id") long requestId;
        @Label("Failed") boolean failed;
    }

    @Name("experiments.DistributedWriteBatch")
    @Label("Distributed write batch")
    @Category({"JExperiments", "DistributedThreadPool"})
    @Enabled(false)
    static final class WriteBatchEvent extends Event {
        @Label("Messages") int messages;
    }
}

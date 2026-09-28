package com.lis.threadpool;

import jdk.jfr.Category;
import jdk.jfr.Enabled;
import jdk.jfr.Event;
import jdk.jfr.EventType;
import jdk.jfr.Label;
import jdk.jfr.Name;

final class OrderedThreadPoolJfr {
    private static final EventType SUBMISSION = EventType.getEventType(SubmissionEvent.class);
    private static final EventType COMPLETION = EventType.getEventType(CompletionEvent.class);
    private static final EventType PUBLICATION = EventType.getEventType(PublicationEvent.class);
    private static final EventType HEAD_OF_LINE = EventType.getEventType(HeadOfLineEvent.class);
    private static final EventType HIGH_WATERMARK = EventType.getEventType(BufferHighWatermarkEvent.class);
    private static final EventType PUBLISHER_BLOCKED = EventType.getEventType(PublisherBlockedEvent.class);

    private OrderedThreadPoolJfr() {
    }

    static void submission(long sequence, boolean hasFuture) {
        if (SUBMISSION.isEnabled()) {
            var event = new SubmissionEvent();
            event.sequence = sequence;
            event.hasFuture = hasFuture;
            event.commit();
        }
    }

    static void completion(long sequence, boolean failed) {
        if (COMPLETION.isEnabled()) {
            var event = new CompletionEvent();
            event.sequence = sequence;
            event.failed = failed;
            event.commit();
        }
    }

    static void publication(long sequence, boolean failed) {
        if (PUBLICATION.isEnabled()) {
            var event = new PublicationEvent();
            event.sequence = sequence;
            event.failed = failed;
            event.commit();
        }
    }

    static void headOfLine(long sequence, long buffered) {
        if (HEAD_OF_LINE.isEnabled()) {
            var event = new HeadOfLineEvent();
            event.sequence = sequence;
            event.buffered = buffered;
            event.commit();
        }
    }

    static void highWatermark(long buffered, long nextSequence) {
        if (HIGH_WATERMARK.isEnabled()) {
            var event = new BufferHighWatermarkEvent();
            event.buffered = buffered;
            event.nextSequence = nextSequence;
            event.commit();
        }
    }

    static PublisherBlockedEvent publisherBlocked(long sequence) {
        if (!PUBLISHER_BLOCKED.isEnabled()) {
            return null;
        }
        var event = new PublisherBlockedEvent();
        event.sequence = sequence;
        return event;
    }

    static void commitPublisherBlocked(PublisherBlockedEvent event) {
        if (event != null) {
            event.end();
            if (event.shouldCommit()) {
                event.commit();
            }
        }
    }

    @Name("experiments.OrderedSubmission")
    @Label("Ordered submission")
    @Category({"JExperiments", "OrderedThreadPoolExecutor"})
    @Enabled(false)
    static final class SubmissionEvent extends Event {
        @Label("Sequence") long sequence;
        @Label("Has future") boolean hasFuture;
    }

    @Name("experiments.OrderedCompletion")
    @Label("Ordered worker completion")
    @Category({"JExperiments", "OrderedThreadPoolExecutor"})
    @Enabled(false)
    static final class CompletionEvent extends Event {
        @Label("Sequence") long sequence;
        @Label("Failed") boolean failed;
    }

    @Name("experiments.OrderedPublication")
    @Label("Ordered publication")
    @Category({"JExperiments", "OrderedThreadPoolExecutor"})
    @Enabled(false)
    static final class PublicationEvent extends Event {
        @Label("Sequence") long sequence;
        @Label("Failed") boolean failed;
    }

    @Name("experiments.OrderedHeadOfLineStall")
    @Label("Ordered head-of-line stall")
    @Category({"JExperiments", "OrderedThreadPoolExecutor"})
    @Enabled(false)
    static final class HeadOfLineEvent extends Event {
        @Label("Waiting for sequence") long sequence;
        @Label("Buffered completions") long buffered;
    }

    @Name("experiments.OrderedBufferHighWatermark")
    @Label("Ordered buffer high-water mark")
    @Category({"JExperiments", "OrderedThreadPoolExecutor"})
    @Enabled(false)
    static final class BufferHighWatermarkEvent extends Event {
        @Label("Buffered completions") long buffered;
        @Label("Next sequence") long nextSequence;
    }

    @Name("experiments.OrderedPublisherBlocked")
    @Label("Ordered publisher blocked")
    @Category({"JExperiments", "OrderedThreadPoolExecutor"})
    @Enabled(false)
    static final class PublisherBlockedEvent extends Event {
        @Label("Sequence") long sequence;
    }
}

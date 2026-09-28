package com.lis.threadpool;

import java.util.Arrays;
import java.util.Objects;

final class OrderedCompletionBuffer<E> {
    private final int segmentShift;
    private final int segmentMask;
    private final int segmentSize;
    private final LongSegmentMap<Segment<E>> segments = new LongSegmentMap<>();
    private int size;

    OrderedCompletionBuffer(int segmentSize) {
        if (segmentSize < 16 || Integer.bitCount(segmentSize) != 1) {
            throw new IllegalArgumentException("segmentSize must be a power of two >= 16");
        }
        this.segmentSize = segmentSize;
        segmentShift = Integer.numberOfTrailingZeros(segmentSize);
        segmentMask = segmentSize - 1;
    }

    void put(long sequence, E value) {
        Objects.requireNonNull(value, "value");
        var segmentId = sequence >>> segmentShift;
        var segment = segments.get(segmentId);
        if (segment == null) {
            segment = new Segment<>(segmentSize);
            segments.put(segmentId, segment);
        }
        var slot = (int) (sequence & segmentMask);
        if (segment.values[slot] != null) {
            throw new IllegalStateException("Duplicate completion sequence: " + sequence);
        }
        segment.values[slot] = value;
        segment.count++;
        size++;
    }

    @SuppressWarnings("unchecked")
    E remove(long sequence) {
        var segmentId = sequence >>> segmentShift;
        var segment = segments.get(segmentId);
        if (segment == null) {
            return null;
        }
        var slot = (int) (sequence & segmentMask);
        var value = (E) segment.values[slot];
        if (value == null) {
            return null;
        }
        segment.values[slot] = null;
        segment.count--;
        size--;
        if (segment.count == 0) {
            segments.remove(segmentId);
        }
        return value;
    }

    @SuppressWarnings("unchecked")
    E removeAny() {
        var segment = segments.firstValue();
        if (segment == null) {
            return null;
        }
        for (var index = 0; index < segment.values.length; index++) {
            var value = (E) segment.values[index];
            if (value != null) {
                segment.values[index] = null;
                segment.count--;
                size--;
                if (segment.count == 0) {
                    segments.removeValue(segment);
                }
                return value;
            }
        }
        throw new IllegalStateException("Segment count is inconsistent");
    }

    int size() {
        return size;
    }

    boolean isEmpty() {
        return size == 0;
    }

    private static final class Segment<E> {
        private final Object[] values;
        private int count;

        private Segment(int size) {
            values = new Object[size];
        }
    }

    private static final class LongSegmentMap<E> {
        private static final long EMPTY = -1L;
        private static final long DELETED = -2L;
        private long[] keys = new long[16];
        private Object[] values = new Object[16];
        private int size;
        private int used;

        private LongSegmentMap() {
            Arrays.fill(keys, EMPTY);
        }

        @SuppressWarnings("unchecked")
        private E get(long key) {
            var index = findIndex(key);
            return index < 0 ? null : (E) values[index];
        }

        private void put(long key, E value) {
            if ((used + 1) * 10 >= keys.length * 6) {
                resize(keys.length << 1);
            }
            var mask = keys.length - 1;
            var index = mix(key) & mask;
            var deleted = -1;
            while (true) {
                var existing = keys[index];
                if (existing == EMPTY) {
                    var target = deleted >= 0 ? deleted : index;
                    if (deleted < 0) {
                        used++;
                    }
                    keys[target] = key;
                    values[target] = value;
                    size++;
                    return;
                }
                if (existing == DELETED) {
                    if (deleted < 0) {
                        deleted = index;
                    }
                } else if (existing == key) {
                    values[index] = value;
                    return;
                }
                index = (index + 1) & mask;
            }
        }

        private void remove(long key) {
            var index = findIndex(key);
            if (index >= 0) {
                keys[index] = DELETED;
                values[index] = null;
                size--;
                compactIfNeeded();
            }
        }

        @SuppressWarnings("unchecked")
        private E firstValue() {
            for (var index = 0; index < keys.length; index++) {
                if (keys[index] >= 0) {
                    return (E) values[index];
                }
            }
            return null;
        }

        private void removeValue(E value) {
            for (var index = 0; index < keys.length; index++) {
                if (keys[index] >= 0 && values[index] == value) {
                    keys[index] = DELETED;
                    values[index] = null;
                    size--;
                    compactIfNeeded();
                    return;
                }
            }
        }

        private int findIndex(long key) {
            var mask = keys.length - 1;
            var index = mix(key) & mask;
            while (true) {
                var existing = keys[index];
                if (existing == EMPTY) {
                    return -1;
                }
                if (existing == key) {
                    return index;
                }
                index = (index + 1) & mask;
            }
        }

        private void compactIfNeeded() {
            if (size == 0) {
                Arrays.fill(keys, EMPTY);
                Arrays.fill(values, null);
                used = 0;
            } else if (used > size * 2) {
                resize(keys.length);
            }
        }

        private void resize(int capacity) {
            var oldKeys = keys;
            var oldValues = values;
            keys = new long[capacity];
            values = new Object[capacity];
            Arrays.fill(keys, EMPTY);
            size = 0;
            used = 0;
            for (var index = 0; index < oldKeys.length; index++) {
                if (oldKeys[index] >= 0) {
                    @SuppressWarnings("unchecked")
                    var value = (E) oldValues[index];
                    put(oldKeys[index], value);
                }
            }
        }

        private static int mix(long value) {
            value ^= value >>> 33;
            value *= 0xff51afd7ed558ccdl;
            value ^= value >>> 33;
            value *= 0xc4ceb9fe1a85ec53l;
            value ^= value >>> 33;
            return (int) value;
        }
    }
}

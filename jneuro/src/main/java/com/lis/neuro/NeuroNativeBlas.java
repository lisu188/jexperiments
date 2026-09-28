package com.lis.neuro;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Objects;
import java.util.Optional;

@SuppressWarnings("restricted")
public final class NeuroNativeBlas {
    private static final int CBLAS_ROW_MAJOR = 101;
    private static final int CBLAS_NO_TRANS = 111;
    private static final String[] LIBRARIES = {
            "libopenblas.so.0",
            "libopenblas.so",
            "libblas.so.3",
            "libblas.so",
            "libmkl_rt.so"
    };

    private NeuroNativeBlas() {
    }

    public static Optional<Session> tryCreate(Neuro network) {
        Objects.requireNonNull(network, "network");
        var arena = Arena.ofShared();
        try {
            var resolved = resolveDgemm(arena);
            return Optional.of(new Session(network, arena, resolved));
        } catch (IllegalArgumentException | IllegalStateException | IllegalCallerException | UnsatisfiedLinkError exception) {
            arena.close();
            return Optional.empty();
        }
    }

    public static final class Session implements AutoCloseable {
        private final Arena arena;
        private final MethodHandle dgemm;
        private final String libraryName;
        private final int[] topology;
        private final MemorySegment[] weights;
        private final double[][] biases;
        private final double beta;
        private final Neuro.SigmoidMode sigmoidMode;
        private final int maxWidth;
        private MemorySegment firstBuffer;
        private MemorySegment secondBuffer;
        private int capacity;

        private Session(Neuro network, Arena arena, Resolved resolved) {
            this.arena = arena;
            dgemm = resolved.handle();
            libraryName = resolved.libraryName();
            topology = network.topology();
            var width = 0;
            for (var size : topology) {
                width = Math.max(width, size);
            }
            maxWidth = width;
            beta = network.hyperParameters().beta();
            sigmoidMode = network.hyperParameters().sigmoidMode();
            weights = new MemorySegment[topology.length - 1];
            biases = new double[topology.length - 1][];
            for (int layer = 0; layer < weights.length; layer++) {
                var inputs = topology[layer];
                var outputs = topology[layer + 1];
                var source = network.backendWeights(layer);
                var transposed = new double[source.length];
                for (int output = 0; output < outputs; output++) {
                    for (int input = 0; input < inputs; input++) {
                        transposed[input * outputs + output] = source[output * inputs + input];
                    }
                }
                var nativeWeights = arena.allocate(ValueLayout.JAVA_DOUBLE, transposed.length);
                MemorySegment.copy(
                        transposed,
                        0,
                        nativeWeights,
                        ValueLayout.JAVA_DOUBLE,
                        0,
                        transposed.length);
                weights[layer] = nativeWeights;
                biases[layer] = network.backendBiases(layer);
            }
        }

        public String libraryName() {
            return libraryName;
        }

        public void predictBatch(double[] inputs, int batchSize, double[] outputs) {
            Objects.requireNonNull(inputs, "inputs");
            Objects.requireNonNull(outputs, "outputs");
            if (batchSize < 0) {
                throw new IllegalArgumentException("batchSize must be >= 0");
            }
            var inputSize = topology[0];
            var outputSize = topology[topology.length - 1];
            var inputElements = Math.multiplyExact(batchSize, inputSize);
            var outputElements = Math.multiplyExact(batchSize, outputSize);
            if (inputs.length < inputElements || outputs.length < outputElements) {
                throw new IllegalArgumentException("batch arrays are too small");
            }
            if (batchSize == 0) {
                return;
            }

            ensureCapacity(batchSize);
            MemorySegment.copy(inputs, 0, firstBuffer, ValueLayout.JAVA_DOUBLE, 0, inputElements);
            var current = firstBuffer;
            var next = secondBuffer;

            for (int layer = 0; layer < weights.length; layer++) {
                var layerInputs = topology[layer];
                var layerOutputs = topology[layer + 1];
                dgemm(
                        batchSize,
                        layerOutputs,
                        layerInputs,
                        current,
                        weights[layer],
                        next);
                activate(next, biases[layer], batchSize, layerOutputs);
                var swap = current;
                current = next;
                next = swap;
            }

            MemorySegment.copy(
                    current,
                    ValueLayout.JAVA_DOUBLE,
                    0,
                    outputs,
                    0,
                    outputElements);
        }

        private void ensureCapacity(int batchSize) {
            if (batchSize <= capacity) {
                return;
            }
            capacity = Math.max(batchSize, Math.max(8, capacity * 2));
            var elements = Math.multiplyExact(capacity, maxWidth);
            firstBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, elements);
            secondBuffer = arena.allocate(ValueLayout.JAVA_DOUBLE, elements);
        }

        private void dgemm(
                int rows,
                int columns,
                int inner,
                MemorySegment left,
                MemorySegment right,
                MemorySegment destination) {
            try {
                dgemm.invoke(
                        CBLAS_ROW_MAJOR,
                        CBLAS_NO_TRANS,
                        CBLAS_NO_TRANS,
                        rows,
                        columns,
                        inner,
                        1.0,
                        left,
                        inner,
                        right,
                        columns,
                        0.0,
                        destination,
                        columns);
            } catch (Throwable throwable) {
                throw new IllegalStateException("cblas_dgemm failed", throwable);
            }
        }

        private void activate(MemorySegment values, double[] layerBiases, int batchSize, int width) {
            for (int sample = 0; sample < batchSize; sample++) {
                var offset = sample * width;
                for (int output = 0; output < width; output++) {
                    var index = offset + output;
                    var value = values.getAtIndex(ValueLayout.JAVA_DOUBLE, index) + layerBiases[output];
                    values.setAtIndex(
                            ValueLayout.JAVA_DOUBLE,
                            index,
                            NeuroNativeBlas.activate(value * beta, sigmoidMode));
                }
            }
        }

        @Override
        public void close() {
            arena.close();
        }
    }

    private record Resolved(MethodHandle handle, String libraryName) {
    }

    private static Resolved resolveDgemm(Arena arena) {
        var linker = Linker.nativeLinker();
        RuntimeException lastFailure = null;
        for (var library : LIBRARIES) {
            try {
                SymbolLookup lookup = SymbolLookup.libraryLookup(library, arena);
                var symbol = lookup.find("cblas_dgemm");
                if (symbol.isPresent()) {
                    var descriptor = FunctionDescriptor.ofVoid(
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_DOUBLE,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_DOUBLE,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT);
                    return new Resolved(linker.downcallHandle(symbol.orElseThrow(), descriptor), library);
                }
            } catch (IllegalArgumentException exception) {
                lastFailure = exception;
            }
        }
        throw new IllegalStateException("No CBLAS implementation with cblas_dgemm found", lastFailure);
    }

    private static double activate(double value, Neuro.SigmoidMode mode) {
        if (mode == Neuro.SigmoidMode.FAST) {
            if (value >= 0.0) {
                var exp = fastExpNegative(-value);
                return 1.0 / (1.0 + exp);
            }
            var exp = fastExpNegative(value);
            return exp / (1.0 + exp);
        }
        return 1.0 / (1.0 + Math.exp(-value));
    }

    private static double fastExpNegative(double value) {
        if (value <= -745.0) {
            return 0.0;
        }
        var exponent = (int) (value * 1.4426950408889634);
        var remainder = value - exponent * 0.6931471805599453;
        var square = remainder * remainder;
        var polynomial = 1.0
                + remainder
                + square * (0.5
                + remainder * (0.16666666666666666
                + remainder * (0.041666666666666664
                + remainder * 0.008333333333333333)));
        return Math.scalb(polynomial, exponent);
    }
}

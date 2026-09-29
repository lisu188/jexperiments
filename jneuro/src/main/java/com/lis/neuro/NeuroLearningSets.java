package com.lis.neuro;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

final class NeuroLearningSets {
    enum Kind {
        XOR("XOR"),
        AND("AND"),
        OR("OR"),
        NAND("NAND"),
        XNOR("XNOR"),
        NOISY_XOR("Noisy XOR"),
        CIRCLE("Circle"),
        SPIRAL("Spiral"),
        CUSTOM("Custom");

        private final String label;

        Kind(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    record Sample(double x, double y, double target) {
        Sample {
            if (!Double.isFinite(x) || !Double.isFinite(y) || !Double.isFinite(target)) {
                throw new IllegalArgumentException("sample values must be finite");
            }
            if (x < 0.0 || x > 1.0 || y < 0.0 || y > 1.0) {
                throw new IllegalArgumentException("sample coordinates must be in [0, 1]");
            }
            if (target < 0.0 || target > 1.0) {
                throw new IllegalArgumentException("target must be in [0, 1]");
            }
        }
    }

    private NeuroLearningSets() {
    }

    static List<Sample> create(Kind kind, long seed) {
        Objects.requireNonNull(kind, "kind");
        return switch (kind) {
            case XOR -> truthTable(0, 1, 1, 0);
            case AND -> truthTable(0, 0, 0, 1);
            case OR -> truthTable(0, 1, 1, 1);
            case NAND -> truthTable(1, 1, 1, 0);
            case XNOR -> truthTable(1, 0, 0, 1);
            case NOISY_XOR -> noisyXor(seed, 160, 0.115);
            case CIRCLE -> circle(seed, 180);
            case SPIRAL -> spiral(220);
            case CUSTOM -> new ArrayList<>();
        };
    }

    static void addTo(Neuro network, List<Sample> samples) {
        Objects.requireNonNull(network, "network");
        Objects.requireNonNull(samples, "samples");
        for (var sample : samples) {
            network.addTrainingSample(
                    new double[]{sample.x(), sample.y()},
                    new double[]{sample.target()});
        }
    }

    static List<Sample> truthTable(int value00, int value01, int value10, int value11) {
        return List.of(
                new Sample(0.0, 0.0, value00),
                new Sample(0.0, 1.0, value01),
                new Sample(1.0, 0.0, value10),
                new Sample(1.0, 1.0, value11));
    }

    private static List<Sample> noisyXor(long seed, int count, double sigma) {
        var random = new SplittableRandom(seed);
        var result = new ArrayList<Sample>(count);
        for (int index = 0; index < count; index++) {
            var corner = index & 3;
            var baseX = corner >= 2 ? 1.0 : 0.0;
            var baseY = (corner & 1) == 1 ? 1.0 : 0.0;
            var target = (corner == 1 || corner == 2) ? 1.0 : 0.0;
            var x = clamp01(baseX + gaussian(random) * sigma);
            var y = clamp01(baseY + gaussian(random) * sigma);
            result.add(new Sample(x, y, target));
        }
        return List.copyOf(result);
    }

    private static List<Sample> circle(long seed, int count) {
        var random = new SplittableRandom(seed);
        var result = new ArrayList<Sample>(count);
        for (int index = 0; index < count; index++) {
            var x = random.nextDouble();
            var y = random.nextDouble();
            var dx = x - 0.5;
            var dy = y - 0.5;
            var target = dx * dx + dy * dy <= 0.26 * 0.26 ? 1.0 : 0.0;
            result.add(new Sample(x, y, target));
        }
        return List.copyOf(result);
    }

    private static List<Sample> spiral(int count) {
        var result = new ArrayList<Sample>(count);
        var perClass = count / 2;
        for (int target = 0; target <= 1; target++) {
            var phase = target * Math.PI;
            for (int index = 0; index < perClass; index++) {
                var fraction = index / (double) Math.max(1, perClass - 1);
                var radius = 0.05 + 0.43 * fraction;
                var angle = phase + fraction * Math.PI * 3.25;
                var x = clamp01(0.5 + radius * Math.cos(angle));
                var y = clamp01(0.5 + radius * Math.sin(angle));
                result.add(new Sample(x, y, target));
            }
        }
        return List.copyOf(result);
    }

    private static double gaussian(SplittableRandom random) {
        var u1 = Math.max(Double.MIN_NORMAL, random.nextDouble());
        var u2 = random.nextDouble();
        return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    }

    private static double clamp01(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }
}

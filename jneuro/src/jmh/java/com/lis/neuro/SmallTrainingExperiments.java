package com.lis.neuro;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;

/** Bounded command-line cohort and paired-seed quality experiments; deliberately outside application code. */
public final class SmallTrainingExperiments {
    private SmallTrainingExperiments() {}
    public static void main(String[] arguments) throws Exception {
        var options = new java.util.HashMap<String, String>();
        for (int i = 0; i < arguments.length; i += 2) {
            if (i + 1 == arguments.length || !arguments[i].startsWith("--")) throw new IllegalArgumentException("Expected --option value");
            options.put(arguments[i].substring(2), arguments[i + 1]);
        }
        String mode = options.getOrDefault("mode", "cohort");
        if (!mode.equals("cohort") && !mode.equals("quality")) throw new IllegalArgumentException("Mode must be cohort or quality");
        Path output = Path.of(options.getOrDefault("output", "build/reports/small-experiments/" + mode + ".csv"));
        Files.createDirectories(output.toAbsolutePath().getParent());
        List<String> rows = mode.equals("quality") ? quality(options) : cohort(options);
        Files.write(output, rows, StandardCharsets.UTF_8);
        Files.writeString(Path.of(output + ".environment.txt"), "java=" + System.getProperty("java.runtime.version") +
            "\nos=" + System.getProperty("os.name") + "\ncpu=" + System.getenv("PROCESSOR_IDENTIFIER") +
            "\nrevision=" + System.getProperty("jneuro.benchmark.sourceRevision", "unspecified") +
            "\nprocessId=" + ProcessHandle.current().pid() + "\njvmStartMillis=" + java.lang.management.ManagementFactory.getRuntimeMXBean().getStartTime() +
            "\njvmArguments=" + java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments() +
            "\nbenchmarkClassSha256=" + classHash() + "\nruntime=TensorFlow " + org.tensorflow.TensorFlow.version() +
            "\noptions=" + options + "\n", StandardCharsets.UTF_8);
        System.out.println("Reports: " + output.toAbsolutePath());
    }

    private static String classHash() throws Exception {
        try (var stream = SmallTrainingExperiments.class.getResourceAsStream("SmallTrainingExperiments.class")) {
            return java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(stream.readAllBytes()));
        }
    }

    private static int integer(java.util.Map<String, String> options, String name, int fallback, int maximum) {
        int value = Integer.parseInt(options.getOrDefault(name, Integer.toString(fallback)));
        if (value < 1 || value > maximum) throw new IllegalArgumentException(name + " must be 1.." + maximum);
        return value;
    }

    private static Neuro model(int[] shape, long seed, Neuro.SigmoidMode sigmoid, int samples, boolean xor) {
        Neuro result = new Neuro(shape, new Neuro.HyperParameters(xor ? 0.5 : 0.05, xor ? 0.2 : 0.1,
            1.0, seed).withSigmoidMode(sigmoid));
        for (int sample = 0; sample < samples; sample++) {
            double x = xor ? ((sample & 3) / 2) : ((sample * 17) & 255) / 255.0;
            double y = xor ? (sample & 1) : ((sample * 17 + 13) & 255) / 255.0;
            double target = xor ? ((sample & 3) == 1 || (sample & 3) == 2 ? 1 : 0) : (sample & 1);
            result.addTrainingSample(new double[] {x, y}, new double[] {target});
        }
        return result;
    }

    private static TrainingChunkRequest request(int epochs) {
        return new TrainingChunkRequest(epochs, null, 1, () -> false, 25_000_000L);
    }

    private static void train(NeuroTrainingSession session, int epochs) {
        int completed = 0;
        while (completed < epochs) {
            int count = session.trainChunk(request(epochs - completed)).getCommittedEpochs();
            if (count < 1) throw new IllegalStateException("No training progress");
            completed += count;
        }
    }

    private static List<String> quality(java.util.Map<String, String> options) throws Exception {
        int seeds = integer(options, "seeds", 32, 1024);
        int epochs = integer(options, "epochs", 10000, 1000000);
        int checkEvery = integer(options, "check-every", 25, 10000);
        double target = Double.parseDouble(options.getOrDefault("target", "0.05"));
        if (!Double.isFinite(target) || target <= 0) throw new IllegalArgumentException("Target must be positive and finite");
        List<String> rows = new ArrayList<>();
        rows.add("seed,precision,sigmoid,epochs,rmse,converged,nanos,target,check_every");
        // Paired identical seeds/data; rotate all four math choices to avoid a fixed within-seed order.
        for (int seed = -2; seed < seeds; seed++) for (int offset = 0; offset < 4; offset++) {
            int variant = Math.floorMod(seed + offset, 4);
            Neuro.TrainingPrecision precision = variant < 2 ? Neuro.TrainingPrecision.FP64 : Neuro.TrainingPrecision.FP32;
            Neuro.SigmoidMode sigmoid = (variant & 1) == 0 ? Neuro.SigmoidMode.EXACT : Neuro.SigmoidMode.FAST;
            Neuro network = model(new int[] {2, 8, 8, 8, 1}, Math.max(0, seed), sigmoid, 4, true);
            int completed = 0;
            double error = network.trainingError();
            long started;
            long elapsed;
            try (NeuroTrainingSession session = network.newTrainingSession(TrainingBackend.CPU, precision, 1, TrainingEngine.SMALL)) {
                started = System.nanoTime();
                int limit = seed < 0 ? Math.min(1000, epochs) : epochs;
                while (completed < limit && error > target) {
                    int count = Math.min(checkEvery, limit - completed);
                    train(session, count);
                    completed += count;
                    error = network.trainingError();
                }
                elapsed = System.nanoTime() - started;
            }
            if (seed >= 0) {
                rows.add(String.format(Locale.ROOT, "%d,%s,%s,%d,%.17g,%s,%d,%.17g,%d", seed, precision, sigmoid,
                    completed, error, error <= target, elapsed, target, checkEvery));
                System.out.printf(Locale.ROOT, "seed=%d %s/%s epochs=%d RMSE=%.6f converged=%s time=%.3fms%n",
                    seed, precision, sigmoid, completed, error, error <= target, elapsed / 1e6);
            }
        }
        return rows;
    }

    private static List<String> cohort(java.util.Map<String, String> options) throws Exception {
        int count = integer(options, "models", 32, 4096);
        int epochs = integer(options, "epochs", 64, 10000);
        int samples = integer(options, "samples", 128, 8192);
        int batch = integer(options, "batch", 64, 8192);
        int workers = integer(options, "parallelism", 4, 256);
        int repeats = integer(options, "repeats", 8, 1000);
        int warmups = integer(options, "warmups", 3, 100);
        int[] shape = Arrays.stream(options.getOrDefault("topology", "2,8,8,8,1").split(",")).mapToInt(Integer::parseInt).toArray();
        String[] engines = options.getOrDefault("engines", "CPU_SESSION_CALL,CPU_PARALLEL_CALL,CPU_SESSION_RETAINED,CPU_COHORT").split(",");
        Neuro.TrainingPrecision precision = Neuro.TrainingPrecision.valueOf(options.getOrDefault("precision", "FP64"));
        if (!SmallNetworkShape.INSTANCE.supports(shape)) throw new IllegalArgumentException("Unsupported small-network topology");
        var supported = java.util.Set.of("CPU_SESSION_CALL", "CPU_PARALLEL_CALL", "CPU_SESSION_RETAINED", "CPU_COHORT", "GPU_COHORT");
        if (engines.length == 0 || Arrays.stream(engines).anyMatch(engine -> !supported.contains(engine)))
            throw new IllegalArgumentException("Unknown TensorFlow execution mode");
        var expected = new ArrayList<Neuro>();
        for (int index = 0; index < count; index++) {
            Neuro reference = model(shape, 1234L + index, Neuro.SigmoidMode.EXACT, samples, false);
            referenceTrain(reference, epochs, batch, precision);
            expected.add(reference);
        }
        List<String> rows = new ArrayList<>();
        rows.add("round,engine,models,precision,epochs,batch,workers,open_ns,training_ns,close_ns,total_ns,max_scaled_error,cpu_worker_limit,actual_precision,actual_backend,kernel");
        try (NeuroTrainingDeviceService service = new NeuroTrainingDeviceService(workers)) {
            for (int round = -warmups; round < repeats; round++) for (int offset = 0; offset < engines.length; offset++) {
                String engine = engines[Math.floorMod(round + offset, engines.length)];
                List<Neuro> networks = new ArrayList<>();
                for (int index = 0; index < count; index++) networks.add(model(shape, 1234L + index, Neuro.SigmoidMode.EXACT, samples, false));
                long started = System.nanoTime();
                NeuroTrainingCohort group = engine.equals("CPU_COHORT") || engine.equals("GPU_COHORT") ?
                    service.openCohort(networks, engine.equals("GPU_COHORT") ? TrainingBackend.CUDA : TrainingBackend.CPU, precision, batch, workers) : null;
                List<NeuroTrainingSession> sessions = new ArrayList<>();
                long opened;
                long trained;
                try {
                    if (engine.equals("CPU_SESSION_RETAINED")) for (Neuro network : networks)
                        sessions.add(service.openSession(network, TrainingBackend.CPU, precision, batch, TrainingEngine.SMALL));
                    opened = System.nanoTime();
                    if (group != null) {
                        int completed = 0;
                        boolean[] active = new boolean[count];
                        Arrays.fill(active, true);
                        while (completed < epochs) {
                            int advanced = group.trainChunk(request(epochs - completed), active).getFirst().getCommittedEpochs();
                            if (advanced < 1) throw new IllegalStateException("No cohort progress");
                            completed += advanced;
                        }
                    } else if (!sessions.isEmpty()) for (NeuroTrainingSession session : sessions) train(session, epochs);
                    else if (engine.equals("CPU_SESSION_CALL")) for (Neuro network : networks) referenceTrain(network, epochs, batch, precision);
                    else if (engine.equals("CPU_PARALLEL_CALL")) {
                        try (var pool = Executors.newFixedThreadPool(workers)) {
                            List<Callable<Void>> tasks = networks.stream().<Callable<Void>>map(network -> () -> { referenceTrain(network, epochs, batch, precision); return null; }).toList();
                            for (var result : pool.invokeAll(tasks)) result.get();
                        }
                    } else throw new IllegalArgumentException("Unknown engine " + engine);
                    trained = System.nanoTime();
                } finally {
                    for (NeuroTrainingSession session : sessions) session.close();
                    if (group != null) group.close();
                }
                long closed = System.nanoTime();
                double scaled = 0;
                for (int index = 0; index < count; index++) {
                    Neuro reference = expected.get(index), actual = networks.get(index);
                    if (actual.statistics().epochsTrained() != epochs || actual.statistics().samplesSeen() != (long) epochs * samples)
                        throw new IllegalStateException("Wrong published epoch/sample count");
                    var validation = NeuroCudaBenchmarkHarness.INSTANCE.validate$experiments_JNeuro(
                        reference.exportTrainingState$experiments_JNeuro(), actual.exportTrainingState$experiments_JNeuro(),
                        reference.trainingError(), actual.trainingError(), precision);
                    scaled = Math.max(scaled, validation.getMaximumScaledError());
                }
                if (scaled > 1.0) throw new IllegalStateException("Full cohort state outside parity tolerance: " + scaled);
                int actualWorkers = switch (engine) {
                    case "CPU_COHORT" -> Math.min(workers, count);
                    case "CPU_PARALLEL_CALL" -> Math.min(workers, count);
                    case "GPU_COHORT" -> 0;
                    default -> 1;
                };
                TrainingDeviceInfo device = group != null ? group.getInfo() : !sessions.isEmpty() ? sessions.getFirst().getInfo() :
                    TensorFlowMath.INSTANCE.info(TrainingBackend.CPU, precision, TrainingEngine.SMALL, Neuro.SigmoidMode.EXACT);
                if (round >= 0) rows.add(String.format(Locale.ROOT, "%d,%s,%d,%s,%d,%d,%d,%d,%d,%d,%d,%.17g,%d,%s,%s,%s",
                    round, engine, count, precision, epochs, batch, workers, opened - started, trained - opened, closed - trained, closed - started, scaled, actualWorkers, device.getPrecision(), device.getBackend(), device.getKernelVersion()));
                System.out.printf(Locale.ROOT, "round=%d %s models=%d train=%.3fms total=%.3fms scaledError=%.3g%n",
                    round, engine, count, (trained - opened) / 1e6, (closed - started) / 1e6, scaled);
            }
        }
        return rows;
    }

    private static void referenceTrain(Neuro network, int epochs, int batch, Neuro.TrainingPrecision precision) {
        try (NeuroTrainingSession session = network.newTrainingSession(TrainingBackend.CPU, precision, batch, TrainingEngine.SMALL)) {
            train(session, epochs);
        }
    }
}

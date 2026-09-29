package com.lis.neuro;

import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

@SuppressWarnings("serial")
public final class NeuroXorCanvas extends Canvas {
    private static final long serialVersionUID = 1L;
    private static final int OUTPUT_GRID_SIZE = 256;
    private static final int HIDDEN_GRID_SIZE = 96;
    private static final int STEP_GRID_SIZE = 192;
    private static final int SEED_GRID_SIZE = 150;
    private static final int MAX_EPOCHS = 10_000;
    private static final double TARGET_ERROR = 0.05;
    private static final long DATASET_SEED = 0xC0FFEE42L;
    private static final int[] TIMELINE_TARGETS = {0, 10, 50, 100, 250, 500, 1_000, 2_000, 5_000, 10_000};
    private static final long[] STUDY_SEEDS = {1L, 42L, 123L, 999L};
    private static final Color BACKGROUND = new Color(22, 24, 29);
    private static final Color PANEL = new Color(43, 47, 55);
    private static final Color FOREGROUND = new Color(220, 224, 230);
    private static final Color MUTED = new Color(132, 140, 150);
    private static final Color POSITIVE = new Color(65, 180, 255);
    private static final Color NEGATIVE = new Color(255, 126, 75);
    private static final Color BOUNDARY = new Color(255, 72, 72);
    private static final Color RMSE_COLOR = new Color(255, 215, 70);
    private static final Color OUTPUT_00_COLOR = new Color(70, 180, 255);
    private static final Color OUTPUT_01_COLOR = new Color(90, 220, 130);
    private static final Color OUTPUT_10_COLOR = new Color(210, 110, 255);
    private static final Color OUTPUT_11_COLOR = new Color(255, 145, 70);
    private static final Color[] SERIES_COLORS = {
        new Color(78, 205, 196),
        new Color(255, 107, 107),
        new Color(255, 209, 102),
        new Color(92, 128, 188),
        new Color(199, 125, 255),
        new Color(100, 220, 120),
        new Color(255, 150, 90),
        new Color(95, 190, 255),
        new Color(244, 114, 182),
        new Color(180, 180, 180),
        new Color(120, 220, 210),
        new Color(240, 220, 100)
    };

    private enum ViewMode {
        OVERVIEW("Overview"),
        DATASET("Learning set"),
        STEP_EFFECT("Step effect"),
        PARAMETERS("Parameters"),
        SEEDS("Seeds"),
        TIMELINE("Timeline");

        private final String label;

        ViewMode(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    private record HistoryPoint(
            int epoch,
            double error,
            double output00,
            double output01,
            double output10,
            double output11,
            double weightNorm0,
            double biasNorm0,
            double weightNorm1,
            double biasNorm1,
            double[] parameters) {
        HistoryPoint {
            parameters = parameters.clone();
        }

        @Override
        public double[] parameters() {
            return parameters.clone();
        }

        double parameter(int index) {
            return parameters[index];
        }
    }

    private record TimelineFrame(int epoch, BufferedImage image) {
    }

    private record SeedResult(long seed, int epochs, double error, BufferedImage image) {
    }

    private record FrameSnapshot(
            BufferedImage outputImage,
            BufferedImage beforeImage,
            BufferedImage differenceImage,
            BufferedImage[] hiddenImages,
            NeuroXorDiagnostics.Snapshot diagnostics,
            List<HistoryPoint> history,
            List<TimelineFrame> timeline,
            List<SeedResult> seedResults,
            List<NeuroLearningSets.Sample> samples,
            NeuroLearningSets.Kind dataset,
            boolean converged) {
        FrameSnapshot {
            hiddenImages = hiddenImages.clone();
            history = List.copyOf(history);
            timeline = List.copyOf(timeline);
            seedResults = List.copyOf(seedResults);
            samples = List.copyOf(samples);
        }

        @Override
        public BufferedImage[] hiddenImages() {
            return hiddenImages.clone();
        }
    }

    private final Object trainingLock = new Object();
    private final double[] gridInputs = NeuroXorGrid.createInputs(OUTPUT_GRID_SIZE);
    private final double[] gridOutputs = new double[OUTPUT_GRID_SIZE * OUTPUT_GRID_SIZE];
    private final Rectangle interactivePlotBounds = new Rectangle();
    private final ArrayList<NeuroLearningSets.Sample> customSamples = new ArrayList<>();
    private final ArrayList<HistoryPoint> history = new ArrayList<>();
    private final ArrayList<TimelineFrame> timeline = new ArrayList<>();

    private volatile FrameSnapshot snapshot;
    private volatile boolean running = true;
    private volatile boolean paused;
    private volatile ViewMode viewMode = ViewMode.OVERVIEW;
    private volatile NeuroLearningSets.Kind selectedDataset = NeuroLearningSets.Kind.XOR;
    private volatile int speed = 10;

    private int pendingSteps;
    private boolean resetRequested;
    private Thread trainingThread;
    private JButton pauseButton;
    private JComboBox<NeuroLearningSets.Kind> datasetBox;
    private Neuro network;
    private List<NeuroLearningSets.Sample> trainingSamples = List.of();
    private List<SeedResult> seedResults = List.of();
    private NeuroXorDiagnostics.Snapshot previousDiagnostics;
    private int epoch;
    private double error;
    private int timelineTargetIndex;
    private double hoverX = 0.5;
    private double hoverY = 0.5;

    public NeuroXorCanvas() {
        setPreferredSize(new Dimension(1460, 940));
        setBackground(BACKGROUND);
        resetModelNow(false);
        installMouseInteraction();
    }

    public static void main(String[] args) {
        EventQueue.invokeLater(() -> {
            var canvas = new NeuroXorCanvas();
            canvas.showWindow();
        });
    }

    private void showWindow() {
        var frame = new JFrame("JNeuro visual learning playground");
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        frame.add(createViewControls(), BorderLayout.NORTH);
        frame.add(this, BorderLayout.CENTER);
        frame.add(createTrainingControls(), BorderLayout.SOUTH);
        frame.pack();
        frame.setLocationByPlatform(true);
        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent event) {
                stopTraining();
            }
        });
        frame.setVisible(true);
        startTraining();
    }

    private JPanel createViewControls() {
        var panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 7, 7));
        for (var mode : ViewMode.values()) {
            var button = new JButton(mode.toString());
            button.addActionListener(event -> {
                viewMode = mode;
                repaint();
            });
            panel.add(button);
        }

        panel.add(new JLabel("  Dataset:"));
        datasetBox = new JComboBox<>(NeuroLearningSets.Kind.values());
        datasetBox.setSelectedItem(selectedDataset);
        datasetBox.addActionListener(event -> {
            var selected = (NeuroLearningSets.Kind) datasetBox.getSelectedItem();
            if (selected != null) {
                requestDataset(selected);
            }
        });
        panel.add(datasetBox);

        var clear = new JButton("Clear custom");
        clear.addActionListener(event -> clearCustomSamples());
        panel.add(clear);
        return panel;
    }

    private JPanel createTrainingControls() {
        var controls = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 7));

        pauseButton = new JButton("Pause");
        pauseButton.addActionListener(event -> togglePause());
        controls.add(pauseButton);

        var stepOne = new JButton("Step 1 epoch");
        stepOne.addActionListener(event -> requestSteps(1));
        controls.add(stepOne);

        var stepTen = new JButton("Step 10 epochs");
        stepTen.addActionListener(event -> requestSteps(10));
        controls.add(stepTen);

        var reset = new JButton("Reset model");
        reset.addActionListener(event -> requestReset());
        controls.add(reset);

        controls.add(new JLabel("Speed:"));
        for (var value : new int[]{1, 10, 100}) {
            var button = new JButton(value + "x");
            button.addActionListener(event -> speed = value);
            controls.add(button);
        }

        controls.add(new JLabel("Left click custom = 1, right click = 0"));
        return controls;
    }

    private void installMouseInteraction() {
        var mouse = new MouseAdapter() {
            @Override
            public void mouseMoved(MouseEvent event) {
                updateHover(event);
            }

            @Override
            public void mouseDragged(MouseEvent event) {
                updateHover(event);
            }

            @Override
            public void mousePressed(MouseEvent event) {
                if (viewMode != ViewMode.DATASET || selectedDataset != NeuroLearningSets.Kind.CUSTOM) {
                    return;
                }
                if (!interactivePlotBounds.contains(event.getPoint())) {
                    return;
                }

                var x = clamp01((event.getX() - interactivePlotBounds.x) / (double) interactivePlotBounds.width);
                var y = clamp01(1.0 - (event.getY() - interactivePlotBounds.y) / (double) interactivePlotBounds.height);
                var target = event.getButton() == MouseEvent.BUTTON3 ? 0.0 : 1.0;
                synchronized (trainingLock) {
                    customSamples.add(new NeuroLearningSets.Sample(x, y, target));
                    resetRequested = true;
                    pendingSteps = 0;
                    trainingLock.notifyAll();
                }
            }
        };
        addMouseMotionListener(mouse);
        addMouseListener(mouse);
    }

    private void updateHover(MouseEvent event) {
        if (!interactivePlotBounds.contains(event.getPoint())) {
            return;
        }
        hoverX = clamp01((event.getX() - interactivePlotBounds.x) / (double) interactivePlotBounds.width);
        hoverY = clamp01(1.0 - (event.getY() - interactivePlotBounds.y) / (double) interactivePlotBounds.height);
        repaint();
    }

    private void startTraining() {
        trainingThread = Thread.ofPlatform()
                .name("jneuro-visual-learning")
                .daemon()
                .start(this::trainingLoop);
    }

    private void stopTraining() {
        synchronized (trainingLock) {
            running = false;
            trainingLock.notifyAll();
        }
        var thread = trainingThread;
        if (thread != null) {
            thread.interrupt();
        }
    }

    private void togglePause() {
        synchronized (trainingLock) {
            paused = !paused;
            trainingLock.notifyAll();
        }
        updatePauseButton();
    }

    private void requestSteps(int epochs) {
        synchronized (trainingLock) {
            paused = true;
            pendingSteps += epochs;
            trainingLock.notifyAll();
        }
        updatePauseButton();
    }

    private void requestReset() {
        synchronized (trainingLock) {
            resetRequested = true;
            pendingSteps = 0;
            trainingLock.notifyAll();
        }
    }

    private void requestDataset(NeuroLearningSets.Kind kind) {
        synchronized (trainingLock) {
            selectedDataset = kind;
            resetRequested = true;
            pendingSteps = 0;
            trainingLock.notifyAll();
        }
    }

    private void clearCustomSamples() {
        synchronized (trainingLock) {
            customSamples.clear();
            selectedDataset = NeuroLearningSets.Kind.CUSTOM;
            resetRequested = true;
            pendingSteps = 0;
            trainingLock.notifyAll();
        }
        if (datasetBox != null) {
            datasetBox.setSelectedItem(NeuroLearningSets.Kind.CUSTOM);
        }
    }

    private void updatePauseButton() {
        var button = pauseButton;
        if (button != null) {
            button.setText(paused ? "Resume" : "Pause");
        }
    }

    private void trainingLoop() {
        try {
            seedResults = computeSeedStudy(trainingSamples);
            publishSnapshot();
            while (running) {
                if (consumeReset()) {
                    resetModelNow(true);
                    continue;
                }

                var epochs = nextTrainingBatch();
                if (epochs == 0) {
                    synchronized (trainingLock) {
                        if (running && pendingSteps == 0 && !resetRequested
                                && (paused || trainingSamples.isEmpty() || convergedOrFinished())) {
                            trainingLock.wait();
                        }
                    }
                    continue;
                }

                for (int index = 0; index < epochs && !trainingSamples.isEmpty()
                        && epoch < MAX_EPOCHS && !converged(); index++) {
                    error = network.trainEpoch();
                    epoch++;
                }
                publishSnapshot();

                if (convergedOrFinished()) {
                    synchronized (trainingLock) {
                        paused = true;
                    }
                    EventQueue.invokeLater(this::updatePauseButton);
                } else if (!paused) {
                    Thread.sleep(16);
                }
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private int nextTrainingBatch() {
        synchronized (trainingLock) {
            if (resetRequested || trainingSamples.isEmpty()) {
                return 0;
            }
            if (pendingSteps > 0) {
                var result = capAtNextTimeline(Math.min(pendingSteps, 100));
                pendingSteps -= result;
                return result;
            }
            if (!paused && !convergedOrFinished()) {
                return capAtNextTimeline(Math.min(speed, MAX_EPOCHS - epoch));
            }
            return 0;
        }
    }

    private int capAtNextTimeline(int epochs) {
        if (timelineTargetIndex >= TIMELINE_TARGETS.length) {
            return epochs;
        }
        var target = TIMELINE_TARGETS[timelineTargetIndex];
        if (target <= epoch) {
            return epochs;
        }
        return Math.min(epochs, target - epoch);
    }

    private boolean consumeReset() {
        synchronized (trainingLock) {
            if (!resetRequested) {
                return false;
            }
            resetRequested = false;
            return true;
        }
    }

    private void resetModelNow(boolean computeSeeds) {
        var dataset = selectedDataset;
        List<NeuroLearningSets.Sample> samples;
        synchronized (trainingLock) {
            samples = dataset == NeuroLearningSets.Kind.CUSTOM
                    ? List.copyOf(customSamples)
                    : NeuroLearningSets.create(dataset, DATASET_SEED);
        }

        network = createNetwork(42L, samples);
        trainingSamples = samples;
        epoch = 0;
        error = samples.isEmpty() ? Double.NaN : network.trainingError();
        history.clear();
        timeline.clear();
        timelineTargetIndex = 0;
        previousDiagnostics = null;
        seedResults = computeSeeds ? computeSeedStudy(samples) : List.of();
        paused = samples.isEmpty();
        publishSnapshot();
        EventQueue.invokeLater(this::updatePauseButton);
    }

    private static Neuro createNetwork(long seed, List<NeuroLearningSets.Sample> samples) {
        var result = new Neuro(
                new int[]{2, 6, 1},
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.6)
                        .withMomentum(0.2)
                        .withSeed(seed));
        NeuroLearningSets.addTo(result, samples);
        return result;
    }

    private List<SeedResult> computeSeedStudy(List<NeuroLearningSets.Sample> samples) {
        if (samples.isEmpty()) {
            return List.of();
        }
        var epochs = samples.size() <= 8 ? 600 : 180;
        var results = new ArrayList<SeedResult>(STUDY_SEEDS.length);
        for (var seed : STUDY_SEEDS) {
            var study = createNetwork(seed, samples);
            study.train(epochs);
            var studyError = study.trainingError();
            var diagnostics = NeuroXorDiagnostics.capture(study, epochs, studyError);
            results.add(new SeedResult(
                    seed,
                    epochs,
                    studyError,
                    NeuroXorDiagnostics.renderOutputMap(diagnostics, SEED_GRID_SIZE)));
        }
        return List.copyOf(results);
    }

    private void publishSnapshot() {
        var outputImage = NeuroXorGrid.render(network, OUTPUT_GRID_SIZE, gridInputs, gridOutputs);
        var diagnostics = NeuroXorDiagnostics.capture(network, epoch, error);
        var hiddenImages = NeuroXorDiagnostics.renderHiddenMaps(diagnostics, HIDDEN_GRID_SIZE);
        var beforeImage = previousDiagnostics == null
                ? outputImage
                : NeuroXorDiagnostics.renderOutputMap(previousDiagnostics, STEP_GRID_SIZE);
        var differenceImage = previousDiagnostics == null
                ? NeuroXorDiagnostics.renderDifferenceMap(diagnostics, diagnostics, STEP_GRID_SIZE)
                : NeuroXorDiagnostics.renderDifferenceMap(previousDiagnostics, diagnostics, STEP_GRID_SIZE);

        var bottom = (OUTPUT_GRID_SIZE - 1) * OUTPUT_GRID_SIZE;
        var output00 = gridOutputs[bottom];
        var output01 = gridOutputs[0];
        var output10 = gridOutputs[bottom + OUTPUT_GRID_SIZE - 1];
        var output11 = gridOutputs[OUTPUT_GRID_SIZE - 1];

        if (history.isEmpty() || history.get(history.size() - 1).epoch() != epoch) {
            history.add(new HistoryPoint(
                    epoch,
                    error,
                    output00,
                    output01,
                    output10,
                    output11,
                    NeuroXorDiagnostics.weightNorm(diagnostics, 0),
                    NeuroXorDiagnostics.biasNorm(diagnostics, 0),
                    NeuroXorDiagnostics.weightNorm(diagnostics, 1),
                    NeuroXorDiagnostics.biasNorm(diagnostics, 1),
                    diagnostics.parameters()));
        }

        captureTimeline(outputImage);
        snapshot = new FrameSnapshot(
                outputImage,
                beforeImage,
                differenceImage,
                hiddenImages,
                diagnostics,
                history,
                timeline,
                seedResults,
                trainingSamples,
                selectedDataset,
                converged());
        previousDiagnostics = diagnostics;
        EventQueue.invokeLater(this::repaint);
    }

    private void captureTimeline(BufferedImage outputImage) {
        if (timelineTargetIndex >= TIMELINE_TARGETS.length) {
            return;
        }
        if (epoch < TIMELINE_TARGETS[timelineTargetIndex]) {
            return;
        }
        timeline.add(new TimelineFrame(epoch, copyImage(outputImage)));
        timelineTargetIndex++;
    }

    private boolean converged() {
        return Double.isFinite(error) && error <= TARGET_ERROR;
    }

    private boolean convergedOrFinished() {
        return converged() || epoch >= MAX_EPOCHS;
    }

    @Override
    public void update(Graphics graphics) {
        paint(graphics);
    }

    @Override
    public void paint(Graphics graphics) {
        var current = snapshot;
        if (current == null) {
            return;
        }

        var g = (Graphics2D) graphics.create();
        try {
            configureGraphics(g);
            g.setColor(BACKGROUND);
            g.fillRect(0, 0, getWidth(), getHeight());
            interactivePlotBounds.setBounds(0, 0, 0, 0);

            switch (viewMode) {
                case OVERVIEW -> drawOverview(g, current);
                case DATASET -> drawDataset(g, current);
                case STEP_EFFECT -> drawStepEffect(g, current);
                case PARAMETERS -> drawParameters(g, current);
                case SEEDS -> drawSeeds(g, current);
                case TIMELINE -> drawTimeline(g, current);
            }
        } finally {
            g.dispose();
        }
    }

    private void drawOverview(Graphics2D g, FrameSnapshot current) {
        var margin = 28;
        var mainTop = 52;
        var mainSize = Math.min(380, Math.max(280, getHeight() / 2 - 82));
        var outputLeft = 58;
        drawOutputSurface(g, current, outputLeft, mainTop, mainSize, true);

        var networkLeft = outputLeft + mainSize + 95;
        var networkWidth = Math.max(440, getWidth() - networkLeft - margin);
        drawNetwork(g, current, networkLeft, mainTop, networkWidth, mainSize);

        var hiddenTop = mainTop + mainSize + 70;
        var hiddenGap = 12;
        var hiddenSize = Math.min(132, Math.max(70, (getWidth() - 2 * margin - hiddenGap * 5) / 6));
        drawHiddenMaps(g, current, margin, hiddenTop, hiddenSize, hiddenGap);

        var historyTop = hiddenTop + hiddenSize + 66;
        var historyHeight = Math.max(105, getHeight() - historyTop - 20);
        drawHistory(g, current.history(), margin + 30, historyTop, getWidth() - 2 * margin - 30, historyHeight);
    }

    private void drawDataset(Graphics2D g, FrameSnapshot current) {
        drawTitle(g, "Learning set and generalization", 28, 36);
        var size = Math.min(650, getHeight() - 135);
        var left = 70;
        var top = 70;
        drawImagePlot(g, current.outputImage(), left, top, size, "Network output");
        interactivePlotBounds.setBounds(left, top, size, size);
        drawSamples(g, current.samples(), left, top, size, 6);

        var right = left + size + 75;
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 18));
        g.setColor(FOREGROUND);
        g.drawString(current.dataset().toString(), right, top + 8);

        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
        g.setColor(MUTED);
        g.drawString("samples: " + current.samples().size(), right, top + 36);
        g.drawString("epoch: " + current.diagnostics().epoch(), right, top + 58);
        g.drawString(
                "RMSE: " + (Double.isFinite(current.diagnostics().error())
                        ? String.format(Locale.ROOT, "%.6f", current.diagnostics().error())
                        : "n/a"),
                right,
                top + 80);

        g.setColor(POSITIVE);
        g.fillOval(right, top + 112, 12, 12);
        g.setColor(FOREGROUND);
        g.drawString("target 1", right + 21, top + 123);
        g.setColor(NEGATIVE);
        g.fillOval(right, top + 141, 12, 12);
        g.setColor(FOREGROUND);
        g.drawString("target 0", right + 21, top + 152);

        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 14));
        g.setColor(MUTED);
        var textY = top + 200;
        if (current.dataset() == NeuroLearningSets.Kind.CUSTOM) {
            g.drawString("Custom playground:", right, textY);
            g.drawString("left click adds class 1", right, textY + 25);
            g.drawString("right click adds class 0", right, textY + 50);
            g.drawString("the model resets after every edit", right, textY + 75);
        } else {
            g.drawString("Choose another dataset above to compare", right, textY);
            g.drawString("linear, noisy and curved learning problems.", right, textY + 25);
        }

        var probe = NeuroXorDiagnostics.probe(current.diagnostics(), hoverX, hoverY);
        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
        g.setColor(FOREGROUND);
        g.drawString(
                String.format(Locale.ROOT, "probe (%.3f, %.3f) -> %.5f", hoverX, hoverY, probe.output()),
                right,
                textY + 125);
    }

    private void drawStepEffect(Graphics2D g, FrameSnapshot current) {
        drawTitle(g, "Effect of the last training update batch", 28, 36);
        var gap = 28;
        var size = Math.min(360, (getWidth() - 2 * 46 - gap * 2) / 3);
        var top = 95;
        var left = 46;

        drawImagePlot(g, current.beforeImage(), left, top, size, "Before");
        drawImagePlot(g, current.outputImage(), left + size + gap, top, size, "After");
        drawImagePlot(g, current.differenceImage(), left + (size + gap) * 2, top, size, "Difference  Δf");

        drawSamples(g, current.samples(), left, top, size, 4);
        drawSamples(g, current.samples(), left + size + gap, top, size, 4);

        var textY = top + size + 72;
        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 14));
        g.setColor(FOREGROUND);
        g.drawString(
                "Epoch " + current.diagnostics().epoch()
                        + "    RMSE " + formatError(current.diagnostics().error()),
                left,
                textY);
        g.setColor(MUTED);
        g.drawString("Red Δf raises the network output; blue Δf lowers it.", left, textY + 28);
        g.drawString(
                "Use Pause + Step 1 epoch to isolate a single complete SGD epoch.",
                left,
                textY + 52);
    }

    private void drawParameters(Graphics2D g, FrameSnapshot current) {
        drawTitle(g, "Weight and bias evolution", 28, 36);
        var historyData = current.history();
        if (historyData.isEmpty()) {
            return;
        }

        var margin = 46;
        var gap = 28;
        var width = (getWidth() - 2 * margin - gap) / 2;
        var height = (getHeight() - 125 - gap) / 2;
        var top = 72;

        drawParameterChart(g, historyData, margin, top, width, height, 0, 12, "Input → hidden weights");
        drawParameterChart(g, historyData, margin + width + gap, top, width, height, 18, 6, "Hidden → output weights");
        drawParameterChart(g, historyData, margin, top + height + gap, width, height, 12, 6, "Hidden biases");
        drawNormChart(g, historyData, margin + width + gap, top + height + gap, width, height);
    }

    private void drawSeeds(Graphics2D g, FrameSnapshot current) {
        drawTitle(g, "Different initializations, same architecture", 28, 36);
        var results = current.seedResults();
        if (results.isEmpty()) {
            g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 16));
            g.setColor(MUTED);
            g.drawString("Seed study is not available for an empty learning set.", 50, 85);
            return;
        }

        var cols = 2;
        var gapX = 90;
        var gapY = 62;
        var size = Math.min(330, (getHeight() - 170 - gapY) / 2);
        var totalWidth = cols * size + gapX;
        var startX = Math.max(50, (getWidth() - totalWidth) / 2);
        var startY = 82;

        for (int index = 0; index < results.size(); index++) {
            var row = index / cols;
            var col = index % cols;
            var x = startX + col * (size + gapX);
            var y = startY + row * (size + gapY);
            var result = results.get(index);
            drawImagePlot(g, result.image(), x, y, size, "seed " + result.seed());
            g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
            g.setColor(MUTED);
            g.drawString(
                    "epochs " + result.epochs() + "   RMSE " + formatError(result.error()),
                    x,
                    y + size + 38);
        }
    }

    private void drawTimeline(Graphics2D g, FrameSnapshot current) {
        drawTitle(g, "Training timeline snapshots", 28, 36);
        var frames = current.timeline();
        if (frames.isEmpty()) {
            return;
        }

        var cols = 5;
        var gap = 22;
        var margin = 38;
        var size = Math.min(
                235,
                Math.max(120, (getWidth() - 2 * margin - gap * (cols - 1)) / cols));
        var top = 82;

        for (int index = 0; index < frames.size(); index++) {
            var row = index / cols;
            var col = index % cols;
            var x = margin + col * (size + gap);
            var y = top + row * (size + 55);
            var frame = frames.get(index);
            drawImagePlot(g, frame.image(), x, y, size, "epoch " + frame.epoch());
        }

        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 14));
        g.setColor(MUTED);
        g.drawString(
                "Snapshots preserve the geometry of the learned function at meaningful milestones.",
                margin,
                getHeight() - 25);
    }

    private void drawOutputSurface(
            Graphics2D g,
            FrameSnapshot current,
            int left,
            int top,
            int size,
            boolean interactive) {
        drawImagePlot(g, current.outputImage(), left, top, size, "Network output f(x, y)");
        if (interactive) {
            interactivePlotBounds.setBounds(left, top, size, size);
        }
        drawSamples(g, current.samples(), left, top, size, 4);

        var probeX = left + (int) Math.round(hoverX * size);
        var probeY = top + (int) Math.round((1.0 - hoverY) * size);
        g.setColor(BOUNDARY);
        g.setStroke(new BasicStroke(1.5f));
        g.drawOval(probeX - 5, probeY - 5, 10, 10);
        g.drawLine(probeX - 9, probeY, probeX + 9, probeY);
        g.drawLine(probeX, probeY - 9, probeX, probeY + 9);

        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
        g.setColor(FOREGROUND);
        g.drawString(
                String.format(
                        Locale.ROOT,
                        "%s   epoch %,d   RMSE %s   %s",
                        current.dataset(),
                        current.diagnostics().epoch(),
                        formatError(current.diagnostics().error()),
                        current.converged() ? "converged" : paused ? "paused" : "training"),
                left,
                top + size + 57);
        g.setColor(MUTED);
        g.drawString("Hover to inspect the live forward pass", left, top + size + 78);
    }

    private void drawHiddenMaps(
            Graphics2D g,
            FrameSnapshot current,
            int left,
            int top,
            int size,
            int gap) {
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
        g.setColor(FOREGROUND);
        g.drawString("Hidden neuron activations and z = 0 boundaries", left, top - 21);

        var images = current.hiddenImages();
        for (int neuron = 0; neuron < images.length; neuron++) {
            var x = left + neuron * (size + gap);
            drawHeatmap(g, images[neuron], x, top, size);

            var boundary = NeuroXorDiagnostics.boundary(current.diagnostics(), neuron);
            if (boundary != null) {
                var x1 = x + (int) Math.round(boundary.x1() * size);
                var y1 = top + (int) Math.round((1.0 - boundary.y1()) * size);
                var x2 = x + (int) Math.round(boundary.x2() * size);
                var y2 = top + (int) Math.round((1.0 - boundary.y2()) * size);
                g.setColor(BOUNDARY);
                g.setStroke(new BasicStroke(2.0f));
                g.drawLine(x1, y1, x2, y2);
            }

            g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
            g.setColor(FOREGROUND);
            g.drawString("H" + neuron, x, top + size + 16);
            g.setColor(MUTED);
            g.drawString(
                    String.format(
                            Locale.ROOT,
                            "w %.1f %.1f  b %.1f",
                            current.diagnostics().inputWeight(neuron, 0),
                            current.diagnostics().inputWeight(neuron, 1),
                            current.diagnostics().hiddenBias(neuron)),
                    x,
                    top + size + 31);
        }
    }

    private void drawNetwork(
            Graphics2D g,
            FrameSnapshot current,
            int left,
            int top,
            int width,
            int height) {
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 17));
        g.setColor(FOREGROUND);
        g.drawString("Live forward pass and weights", left, top - 18);

        var diagnostics = current.diagnostics();
        var probe = NeuroXorDiagnostics.probe(diagnostics, hoverX, hoverY);
        var hidden = probe.hidden();
        var contributions = probe.contributions();
        var maxWeight = Math.max(1.0e-9, NeuroXorDiagnostics.maxAbsWeight(diagnostics));

        var inputX = left + 38;
        var hiddenX = left + width / 2;
        var outputX = left + width - 48;
        var inputY0 = top + height / 3;
        var inputY1 = top + 2 * height / 3;
        var outputY = top + height / 2;
        var hiddenTop = top + 28;
        var hiddenBottom = top + height - 92;
        var hiddenStep = (hiddenBottom - hiddenTop) / Math.max(1, diagnostics.hiddenCount() - 1);

        for (int neuron = 0; neuron < diagnostics.hiddenCount(); neuron++) {
            var hiddenY = hiddenTop + neuron * hiddenStep;
            drawWeightEdge(g, inputX, inputY0, hiddenX, hiddenY, diagnostics.inputWeight(neuron, 0), maxWeight);
            drawWeightEdge(g, inputX, inputY1, hiddenX, hiddenY, diagnostics.inputWeight(neuron, 1), maxWeight);
            drawWeightEdge(g, hiddenX, hiddenY, outputX, outputY, diagnostics.outputWeight(neuron), maxWeight);
        }

        drawActivationNode(g, inputX, inputY0, hoverX, "x");
        drawActivationNode(g, inputX, inputY1, hoverY, "y");
        for (int neuron = 0; neuron < diagnostics.hiddenCount(); neuron++) {
            var hiddenY = hiddenTop + neuron * hiddenStep;
            drawActivationNode(g, hiddenX, hiddenY, hidden[neuron], "H" + neuron);
        }
        drawActivationNode(g, outputX, outputY, probe.output(), "out");

        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        g.setColor(FOREGROUND);
        g.drawString(
                String.format(
                        Locale.ROOT,
                        "probe (%.3f, %.3f) -> %.6f    output z = %.4f    bias = %.4f",
                        hoverX,
                        hoverY,
                        probe.output(),
                        probe.outputPreActivation(),
                        diagnostics.outputBias()),
                left,
                top + height - 48);

        var barLeft = left + 18;
        var barTop = top + height - 31;
        var available = Math.max(180, width - 36);
        var barWidth = available / diagnostics.hiddenCount();
        var maxContribution = 1.0e-9;
        for (var contribution : contributions) {
            maxContribution = Math.max(maxContribution, Math.abs(contribution));
        }
        for (int neuron = 0; neuron < contributions.length; neuron++) {
            var center = barLeft + neuron * barWidth + barWidth / 2;
            var length = (int) Math.round((barWidth * 0.38) * Math.abs(contributions[neuron]) / maxContribution);
            g.setColor(contributions[neuron] >= 0.0 ? POSITIVE : NEGATIVE);
            if (contributions[neuron] >= 0.0) {
                g.fillRect(center, barTop - 5, length, 10);
            } else {
                g.fillRect(center - length, barTop - 5, length, 10);
            }
            g.setColor(MUTED);
            g.drawLine(center, barTop - 7, center, barTop + 7);
            g.drawString("H" + neuron, center - 7, barTop + 22);
        }
    }

    private void drawHistory(Graphics2D g, List<HistoryPoint> data, int left, int top, int width, int height) {
        if (height <= 30 || data.isEmpty()) {
            return;
        }

        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
        g.setColor(FOREGROUND);
        g.drawString("Training history", left, top - 13);

        var chartTop = top + 8;
        var chartHeight = height - 35;
        var chartWidth = width - 10;
        drawChartBackground(g, left, chartTop, chartWidth, chartHeight);

        var maxEpoch = Math.max(1, data.get(data.size() - 1).epoch());
        drawHistorySeries(g, data, left, chartTop, chartWidth, chartHeight, maxEpoch, RMSE_COLOR, 0);
        drawHistorySeries(g, data, left, chartTop, chartWidth, chartHeight, maxEpoch, OUTPUT_00_COLOR, 1);
        drawHistorySeries(g, data, left, chartTop, chartWidth, chartHeight, maxEpoch, OUTPUT_01_COLOR, 2);
        drawHistorySeries(g, data, left, chartTop, chartWidth, chartHeight, maxEpoch, OUTPUT_10_COLOR, 3);
        drawHistorySeries(g, data, left, chartTop, chartWidth, chartHeight, maxEpoch, OUTPUT_11_COLOR, 4);

        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        var legendX = left + 8;
        var legendY = chartTop + 15;
        legendItem(g, "RMSE", RMSE_COLOR, legendX, legendY);
        legendItem(g, "00", OUTPUT_00_COLOR, legendX + 70, legendY);
        legendItem(g, "01", OUTPUT_01_COLOR, legendX + 115, legendY);
        legendItem(g, "10", OUTPUT_10_COLOR, legendX + 160, legendY);
        legendItem(g, "11", OUTPUT_11_COLOR, legendX + 205, legendY);
    }

    private void drawParameterChart(
            Graphics2D g,
            List<HistoryPoint> data,
            int left,
            int top,
            int width,
            int height,
            int start,
            int count,
            String title) {
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 15));
        g.setColor(FOREGROUND);
        g.drawString(title, left, top - 10);
        drawChartBackground(g, left, top, width, height);

        var maxEpoch = Math.max(1, data.get(data.size() - 1).epoch());
        var maxAbs = 1.0e-9;
        for (var point : data) {
            var parameters = point.parameters();
            for (int index = 0; index < count; index++) {
                maxAbs = Math.max(maxAbs, Math.abs(parameters[start + index]));
            }
        }

        for (int series = 0; series < count; series++) {
            g.setColor(SERIES_COLORS[series % SERIES_COLORS.length]);
            g.setStroke(new BasicStroke(1.4f));
            var previousX = -1;
            var previousY = -1;
            for (var point : data) {
                var value = point.parameter(start + series);
                var x = left + (int) Math.round(width * point.epoch() / (double) maxEpoch);
                var y = top + height / 2 - (int) Math.round((height * 0.45) * value / maxAbs);
                if (previousX >= 0) {
                    g.drawLine(previousX, previousY, x, y);
                }
                previousX = x;
                previousY = y;
            }
        }

        g.setColor(MUTED);
        g.drawLine(left, top + height / 2, left + width, top + height / 2);
        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 10));
        g.drawString(String.format(Locale.ROOT, "+%.2f", maxAbs), left + 5, top + 12);
        g.drawString(String.format(Locale.ROOT, "-%.2f", maxAbs), left + 5, top + height - 5);
    }

    private void drawNormChart(
            Graphics2D g,
            List<HistoryPoint> data,
            int left,
            int top,
            int width,
            int height) {
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 15));
        g.setColor(FOREGROUND);
        g.drawString("Parameter norms", left, top - 10);
        drawChartBackground(g, left, top, width, height);

        var maxEpoch = Math.max(1, data.get(data.size() - 1).epoch());
        var max = 1.0e-9;
        for (var point : data) {
            max = Math.max(max, point.weightNorm0());
            max = Math.max(max, point.biasNorm0());
            max = Math.max(max, point.weightNorm1());
            max = Math.max(max, point.biasNorm1());
        }

        for (int series = 0; series < 4; series++) {
            var color = SERIES_COLORS[series];
            g.setColor(color);
            g.setStroke(new BasicStroke(1.8f));
            var previousX = -1;
            var previousY = -1;
            for (var point : data) {
                var value = switch (series) {
                    case 0 -> point.weightNorm0();
                    case 1 -> point.biasNorm0();
                    case 2 -> point.weightNorm1();
                    case 3 -> point.biasNorm1();
                    default -> throw new IllegalArgumentException("unknown norm series");
                };
                var x = left + (int) Math.round(width * point.epoch() / (double) maxEpoch);
                var y = top + height - (int) Math.round(height * 0.9 * value / max);
                if (previousX >= 0) {
                    g.drawLine(previousX, previousY, x, y);
                }
                previousX = x;
                previousY = y;
            }
        }

        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        legendItem(g, "||W1||", SERIES_COLORS[0], left + 8, top + 16);
        legendItem(g, "||b1||", SERIES_COLORS[1], left + 88, top + 16);
        legendItem(g, "||W2||", SERIES_COLORS[2], left + 168, top + 16);
        legendItem(g, "||b2||", SERIES_COLORS[3], left + 248, top + 16);
    }

    private static void drawHistorySeries(
            Graphics2D g,
            List<HistoryPoint> history,
            int left,
            int top,
            int width,
            int height,
            int maxEpoch,
            Color color,
            int series) {
        g.setColor(color);
        g.setStroke(new BasicStroke(series == 0 ? 2.2f : 1.6f));
        var previousX = -1;
        var previousY = -1;
        for (var point : history) {
            var x = left + (int) Math.round(width * point.epoch() / (double) maxEpoch);
            var value = switch (series) {
                case 0 -> point.error();
                case 1 -> point.output00();
                case 2 -> point.output01();
                case 3 -> point.output10();
                case 4 -> point.output11();
                default -> throw new IllegalArgumentException("unknown history series");
            };
            if (!Double.isFinite(value)) {
                continue;
            }
            var y = top + (int) Math.round(height * (1.0 - clamp01(value)));
            if (previousX >= 0) {
                g.drawLine(previousX, previousY, x, y);
            }
            previousX = x;
            previousY = y;
        }
    }

    private static void drawImagePlot(
            Graphics2D g,
            BufferedImage image,
            int left,
            int top,
            int size,
            String title) {
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 15));
        g.setColor(FOREGROUND);
        g.drawString(title, left, top - 12);
        drawHeatmap(g, image, left, top, size);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 11));
        g.setColor(MUTED);
        g.drawString("0", left - 2, top + size + 15);
        g.drawString("1", left + size - 4, top + size + 15);
        g.drawString("x", left + size / 2, top + size + 30);
        g.drawString("1", left - 16, top + 4);
        g.drawString("0", left - 16, top + size + 4);
    }

    private static void drawSamples(
            Graphics2D g,
            List<NeuroLearningSets.Sample> samples,
            int left,
            int top,
            int size,
            int radius) {
        for (var sample : samples) {
            var x = left + (int) Math.round(sample.x() * size);
            var y = top + (int) Math.round((1.0 - sample.y()) * size);
            g.setColor(sample.target() >= 0.5 ? POSITIVE : NEGATIVE);
            g.fillOval(x - radius, y - radius, radius * 2, radius * 2);
            g.setColor(FOREGROUND);
            g.setStroke(new BasicStroke(1.0f));
            g.drawOval(x - radius, y - radius, radius * 2, radius * 2);
        }
    }

    private static void drawHeatmap(Graphics2D g, BufferedImage image, int x, int y, int size) {
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(image, x, y, size, size, null);
        g.setColor(MUTED);
        g.drawRect(x, y, size, size);
    }

    private static void drawChartBackground(Graphics2D g, int x, int y, int width, int height) {
        g.setColor(PANEL);
        g.fillRect(x, y, width, height);
        g.setColor(MUTED);
        g.drawRect(x, y, width, height);
        g.setColor(new Color(70, 74, 83));
        g.drawLine(x, y + height / 2, x + width, y + height / 2);
    }

    private static void drawTitle(Graphics2D g, String text, int x, int y) {
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 22));
        g.setColor(FOREGROUND);
        g.drawString(text, x, y);
    }

    private static void configureGraphics(Graphics2D g) {
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
    }

    private static void legendItem(Graphics2D g, String label, Color color, int x, int y) {
        g.setColor(color);
        g.fillRect(x, y - 8, 12, 3);
        g.setColor(FOREGROUND);
        g.drawString(label, x + 17, y - 3);
    }

    private static void drawWeightEdge(
            Graphics2D g,
            int x1,
            int y1,
            int x2,
            int y2,
            double weight,
            double maxWeight) {
        var strength = Math.abs(weight) / maxWeight;
        g.setColor(weight >= 0.0 ? POSITIVE : NEGATIVE);
        g.setStroke(new BasicStroke((float) (0.6 + 4.0 * strength)));
        g.drawLine(x1, y1, x2, y2);
    }

    private static void drawActivationNode(Graphics2D g, int x, int y, double activation, String label) {
        var gray = (int) Math.round(255.0 * clamp01(activation));
        g.setColor(new Color(gray, gray, gray));
        g.fillOval(x - 13, y - 13, 26, 26);
        g.setColor(FOREGROUND);
        g.setStroke(new BasicStroke(1.4f));
        g.drawOval(x - 13, y - 13, 26, 26);
        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        g.drawString(label, x - 10, y - 18);
    }

    private static BufferedImage copyImage(BufferedImage source) {
        var result = new BufferedImage(source.getWidth(), source.getHeight(), BufferedImage.TYPE_INT_RGB);
        result.getGraphics().drawImage(source, 0, 0, null);
        return result;
    }

    private static String formatError(double value) {
        return Double.isFinite(value) ? String.format(Locale.ROOT, "%.6f", value) : "n/a";
    }

    private static double clamp01(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }
}

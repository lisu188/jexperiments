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
import javax.swing.JTextField;
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
            double[] weightNorms,
            double[] biasNorms,
            double[] parameters) {
        HistoryPoint {
            weightNorms = weightNorms.clone();
            biasNorms = biasNorms.clone();
            parameters = parameters.clone();
        }

        @Override
        public double[] weightNorms() {
            return weightNorms.clone();
        }

        @Override
        public double[] biasNorms() {
            return biasNorms.clone();
        }

        @Override
        public double[] parameters() {
            return parameters.clone();
        }

        double weightNorm(int layer) {
            return weightNorms[layer];
        }

        double biasNorm(int layer) {
            return biasNorms[layer];
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
    private volatile int[] topology = {2, 6, 1};

    private int pendingSteps;
    private boolean resetRequested;
    private Thread trainingThread;
    private JButton pauseButton;
    private JComboBox<NeuroLearningSets.Kind> datasetBox;
    private JTextField topologyField;
    private JLabel topologyStatus;
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

        panel.add(new JLabel("  Hidden layers:"));
        topologyField = new JTextField(NeuroTopologySpec.hiddenLayersText(topology), 9);
        topologyField.setToolTipText("Comma-separated hidden layer sizes, e.g. 1, 2, 3,2, 8,4,2. Empty = no hidden layer.");
        topologyField.addActionListener(event -> applyTopologyFromField());
        panel.add(topologyField);

        var applyTopology = new JButton("Apply architecture");
        applyTopology.addActionListener(event -> applyTopologyFromField());
        panel.add(applyTopology);

        topologyStatus = new JLabel();
        updateTopologyStatus(null);
        panel.add(topologyStatus);
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

    private void applyTopologyFromField() {
        var field = topologyField;
        if (field == null) {
            return;
        }

        final int[] parsed;
        try {
            parsed = NeuroTopologySpec.parseHiddenLayers(field.getText());
        } catch (IllegalArgumentException exception) {
            updateTopologyStatus(exception.getMessage());
            return;
        }

        synchronized (trainingLock) {
            topology = parsed;
            resetRequested = true;
            pendingSteps = 0;
            trainingLock.notifyAll();
        }
        field.setText(NeuroTopologySpec.hiddenLayersText(parsed));
        updateTopologyStatus(null);
    }

    private void updateTopologyStatus(String errorMessage) {
        var label = topologyStatus;
        if (label == null) {
            return;
        }
        if (errorMessage != null) {
            label.setForeground(NEGATIVE);
            label.setText(errorMessage);
            return;
        }

        var current = topology;
        label.setForeground(MUTED);
        label.setText(
                NeuroTopologySpec.display(current)
                        + "  (" + NeuroTopologySpec.parameterCount(current) + " params)");
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
            seedResults = computeSeedStudy(trainingSamples, topology.clone());
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

        var currentTopology = topology.clone();
        network = createNetwork(42L, samples, currentTopology);
        trainingSamples = samples;
        epoch = 0;
        error = samples.isEmpty() ? Double.NaN : network.trainingError();
        history.clear();
        timeline.clear();
        timelineTargetIndex = 0;
        previousDiagnostics = null;
        seedResults = computeSeeds ? computeSeedStudy(samples, currentTopology) : List.of();
        paused = samples.isEmpty();
        publishSnapshot();
        EventQueue.invokeLater(() -> {
            updatePauseButton();
            updateTopologyStatus(null);
        });
    }

    private static Neuro createNetwork(
            long seed,
            List<NeuroLearningSets.Sample> samples,
            int[] topology) {
        var result = new Neuro(
                topology,
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.6)
                        .withMomentum(0.2)
                        .withSeed(seed));
        NeuroLearningSets.addTo(result, samples);
        return result;
    }

    private List<SeedResult> computeSeedStudy(
            List<NeuroLearningSets.Sample> samples,
            int[] studyTopology) {
        if (samples.isEmpty()) {
            return List.of();
        }
        var epochs = samples.size() <= 8 ? 600 : 180;
        var results = new ArrayList<SeedResult>(STUDY_SEEDS.length);
        for (var seed : STUDY_SEEDS) {
            var study = createNetwork(seed, samples, studyTopology);
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
            var weightNorms = new double[diagnostics.layerCount()];
            var biasNorms = new double[diagnostics.layerCount()];
            for (int layer = 0; layer < diagnostics.layerCount(); layer++) {
                weightNorms[layer] = NeuroXorDiagnostics.weightNorm(diagnostics, layer);
                biasNorms[layer] = NeuroXorDiagnostics.biasNorm(diagnostics, layer);
            }
            history.add(new HistoryPoint(
                    epoch,
                    error,
                    output00,
                    output01,
                    output10,
                    output11,
                    weightNorms,
                    biasNorms,
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
        var hiddenGap = 10;
        var visibleHidden = Math.min(8, current.diagnostics().hiddenNeuronCount());
        var hiddenSize = visibleHidden == 0
                ? 0
                : Math.min(
                        112,
                        Math.max(
                                56,
                                (getWidth() - 2 * margin - hiddenGap * Math.max(0, visibleHidden - 1))
                                        / visibleHidden));
        drawHiddenMaps(g, current, margin, hiddenTop, hiddenSize, hiddenGap);

        var historyTop = hiddenTop + (visibleHidden == 0 ? 38 : hiddenSize + 58);
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

        var diagnostics = current.diagnostics();
        var margin = 46;
        var gap = 28;
        var width = (getWidth() - 2 * margin - gap) / 2;
        var height = (getHeight() - 125 - gap) / 2;
        var top = 72;

        drawLayerParameterChart(g, historyData, diagnostics, 0, margin, top, width, height);
        var lastLayer = diagnostics.layerCount() - 1;
        if (lastLayer != 0) {
            drawLayerParameterChart(
                    g,
                    historyData,
                    diagnostics,
                    lastLayer,
                    margin + width + gap,
                    top,
                    width,
                    height);
        } else {
            drawArchitectureSummary(g, diagnostics, margin + width + gap, top, width, height);
        }

        if (diagnostics.layerCount() > 2) {
            drawLayerParameterChart(
                    g,
                    historyData,
                    diagnostics,
                    1,
                    margin,
                    top + height + gap,
                    width,
                    height);
        } else {
            drawArchitectureSummary(g, diagnostics, margin, top + height + gap, width, height);
        }
        drawNormChart(g, historyData, diagnostics, margin + width + gap, top + height + gap, width, height);
    }

    private void drawSeeds(Graphics2D g, FrameSnapshot current) {
        drawTitle(
                g,
                "Different initializations — " + NeuroTopologySpec.display(current.diagnostics().topology()),
                28,
                36);
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
                        "%s   %s   epoch %,d   RMSE %s   %s",
                        current.dataset(),
                        NeuroTopologySpec.display(current.diagnostics().topology()),
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
        var diagnostics = current.diagnostics();
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
        g.setColor(FOREGROUND);

        if (diagnostics.hiddenLayerCount() == 0) {
            g.drawString("No hidden layer — direct 2 → 1 logistic model", left, top - 8);
            return;
        }

        g.drawString(
                "Hidden activations — red z = 0 boundary is exact only for layer 1",
                left,
                top - 21);

        var images = current.hiddenImages();
        var visible = Math.min(8, images.length);
        var mapIndex = 0;
        var visibleIndex = 0;
        for (int hiddenLayer = 0;
                hiddenLayer < diagnostics.hiddenLayerCount() && visibleIndex < visible;
                hiddenLayer++) {
            var layerOffset = NeuroXorDiagnostics.hiddenMapOffset(diagnostics, hiddenLayer);
            for (int neuron = 0;
                    neuron < diagnostics.hiddenLayerSize(hiddenLayer) && visibleIndex < visible;
                    neuron++) {
                mapIndex = layerOffset + neuron;
                var x = left + visibleIndex * (size + gap);
                drawHeatmap(g, images[mapIndex], x, top, size);

                if (hiddenLayer == 0) {
                    var boundary = NeuroXorDiagnostics.boundary(diagnostics, neuron);
                    if (boundary != null) {
                        var x1 = x + (int) Math.round(boundary.x1() * size);
                        var y1 = top + (int) Math.round((1.0 - boundary.y1()) * size);
                        var x2 = x + (int) Math.round(boundary.x2() * size);
                        var y2 = top + (int) Math.round((1.0 - boundary.y2()) * size);
                        g.setColor(BOUNDARY);
                        g.setStroke(new BasicStroke(2.0f));
                        g.drawLine(x1, y1, x2, y2);
                    }
                }

                g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 10));
                g.setColor(FOREGROUND);
                g.drawString("L" + (hiddenLayer + 1) + " H" + neuron, x, top + size + 14);
                g.setColor(MUTED);
                g.drawString(
                        "b " + String.format(
                                Locale.ROOT,
                                "%.2f",
                                diagnostics.layerBias(hiddenLayer, neuron)),
                        x,
                        top + size + 28);
                visibleIndex++;
            }
        }

        if (images.length > visible) {
            g.setColor(MUTED);
            g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 12));
            g.drawString(
                    "+" + (images.length - visible) + " more hidden neurons",
                    left + visible * (size + gap),
                    top + Math.max(18, size / 2));
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
        var topologyValues = diagnostics.topology();
        var probe = NeuroXorDiagnostics.probe(diagnostics, hoverX, hoverY);
        var activations = probe.activations();
        var contributions = probe.contributions();
        var maxWeight = Math.max(1.0e-9, NeuroXorDiagnostics.maxAbsWeight(diagnostics));

        var columnCount = topologyValues.length;
        var xStart = left + 32;
        var xEnd = left + width - 38;
        var xStep = columnCount <= 1 ? 0 : (xEnd - xStart) / (columnCount - 1);
        var nodeTop = top + 35;
        var nodeBottom = top + height - 105;
        var maxVisibleNodes = 10;

        for (int layer = 0; layer < diagnostics.layerCount(); layer++) {
            var sourceSize = topologyValues[layer];
            var destinationSize = topologyValues[layer + 1];
            var visibleSource = Math.min(sourceSize, maxVisibleNodes);
            var visibleDestination = Math.min(destinationSize, maxVisibleNodes);
            var sourceX = xStart + layer * xStep;
            var destinationX = xStart + (layer + 1) * xStep;

            for (int destination = 0; destination < visibleDestination; destination++) {
                var destinationY = nodeY(nodeTop, nodeBottom, destination, visibleDestination);
                for (int source = 0; source < visibleSource; source++) {
                    var sourceY = nodeY(nodeTop, nodeBottom, source, visibleSource);
                    drawWeightEdge(
                            g,
                            sourceX,
                            sourceY,
                            destinationX,
                            destinationY,
                            diagnostics.layerWeight(layer, destination, source),
                            maxWeight);
                }
            }
        }

        drawActivationNode(g, xStart, nodeY(nodeTop, nodeBottom, 0, 2), hoverX, "x");
        drawActivationNode(g, xStart, nodeY(nodeTop, nodeBottom, 1, 2), hoverY, "y");

        for (int layer = 0; layer < diagnostics.layerCount(); layer++) {
            var values = activations[layer];
            var visible = Math.min(values.length, maxVisibleNodes);
            var x = xStart + (layer + 1) * xStep;
            for (int neuron = 0; neuron < visible; neuron++) {
                var label = layer == diagnostics.layerCount() - 1
                        ? "out"
                        : "L" + (layer + 1) + ":" + neuron;
                drawActivationNode(
                        g,
                        x,
                        nodeY(nodeTop, nodeBottom, neuron, visible),
                        values[neuron],
                        label);
            }
            if (values.length > visible) {
                g.setColor(MUTED);
                g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 10));
                g.drawString("+" + (values.length - visible), x - 8, nodeBottom + 27);
            }
        }

        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        g.setColor(MUTED);
        for (int column = 0; column < topologyValues.length; column++) {
            var label = column == 0
                    ? "input"
                    : column == topologyValues.length - 1 ? "output" : "hidden " + column;
            g.drawString(label, xStart + column * xStep - 18, top + 12);
        }

        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        g.setColor(FOREGROUND);
        g.drawString(
                String.format(
                        Locale.ROOT,
                        "%s   probe (%.3f, %.3f) → %.6f   output z %.4f",
                        NeuroTopologySpec.display(topologyValues),
                        hoverX,
                        hoverY,
                        probe.output(),
                        probe.outputPreActivation()),
                left,
                top + height - 55);

        var barLeft = left + 18;
        var barTop = top + height - 29;
        var visibleContributions = Math.min(contributions.length, 10);
        var available = Math.max(180, width - 36);
        var barWidth = Math.max(1, available / Math.max(1, visibleContributions));
        var maxContribution = 1.0e-9;
        for (int index = 0; index < visibleContributions; index++) {
            maxContribution = Math.max(maxContribution, Math.abs(contributions[index]));
        }
        for (int index = 0; index < visibleContributions; index++) {
            var center = barLeft + index * barWidth + barWidth / 2;
            var length = (int) Math.round(
                    (barWidth * 0.38) * Math.abs(contributions[index]) / maxContribution);
            g.setColor(contributions[index] >= 0.0 ? POSITIVE : NEGATIVE);
            if (contributions[index] >= 0.0) {
                g.fillRect(center, barTop - 5, length, 10);
            } else {
                g.fillRect(center - length, barTop - 5, length, 10);
            }
            g.setColor(MUTED);
            g.drawLine(center, barTop - 7, center, barTop + 7);
            var label = diagnostics.hiddenLayerCount() == 0
                    ? (index == 0 ? "x" : "y")
                    : "H" + index;
            g.drawString(label, center - 7, barTop + 22);
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

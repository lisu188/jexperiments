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
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

@SuppressWarnings("serial")
public final class NeuroXorCanvas extends Canvas {
    private static final long serialVersionUID = 1L;
    private static final int OUTPUT_GRID_SIZE = 256;
    private static final int HIDDEN_GRID_SIZE = 96;
    private static final int EPOCHS_PER_FRAME = 10;
    private static final int MAX_EPOCHS = 10_000;
    private static final double TARGET_ERROR = 0.05;
    private static final Color BACKGROUND = new Color(22, 24, 29);
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

    private final Object trainingLock = new Object();
    private final double[] gridInputs = NeuroXorGrid.createInputs(OUTPUT_GRID_SIZE);
    private final double[] gridOutputs = new double[OUTPUT_GRID_SIZE * OUTPUT_GRID_SIZE];
    private final Rectangle outputBounds = new Rectangle();
    private volatile FrameSnapshot snapshot;
    private volatile boolean running = true;
    private volatile boolean paused;
    private int pendingSteps;
    private boolean resetRequested;
    private Thread trainingThread;
    private JButton pauseButton;
    private Neuro network;
    private double hoverX = 0.5;
    private double hoverY = 0.5;

    private record HistoryPoint(
            int epoch,
            double error,
            double output00,
            double output01,
            double output10,
            double output11) {
    }

    private record FrameSnapshot(
            BufferedImage outputImage,
            BufferedImage[] hiddenImages,
            NeuroXorDiagnostics.Snapshot diagnostics,
            List<HistoryPoint> history,
            boolean converged) {
        FrameSnapshot {
            hiddenImages = hiddenImages.clone();
            history = List.copyOf(history);
        }

        @Override
        public BufferedImage[] hiddenImages() {
            return hiddenImages.clone();
        }
    }

    public NeuroXorCanvas() {
        setPreferredSize(new Dimension(1400, 940));
        setBackground(BACKGROUND);
        network = createNetwork();
        publishSnapshot(0, network.trainingError(), new ArrayList<>());
        installMouseTracking();
    }

    public static void main(String[] args) {
        EventQueue.invokeLater(() -> {
            var canvas = new NeuroXorCanvas();
            canvas.showWindow();
        });
    }

    private static Neuro createNetwork() {
        var result = new Neuro(
                new int[]{2, 6, 1},
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.6)
                        .withMomentum(0.2)
                        .withSeed(42));
        addXorSamples(result);
        return result;
    }

    private void showWindow() {
        var frame = new JFrame("JNeuro XOR learning dashboard");
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        frame.add(this, BorderLayout.CENTER);
        frame.add(createControls(), BorderLayout.SOUTH);
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

    private JPanel createControls() {
        var controls = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 7));
        pauseButton = new JButton("Pause");
        pauseButton.addActionListener(event -> togglePause());

        var stepOne = new JButton("Step 1 epoch");
        stepOne.addActionListener(event -> requestSteps(1));

        var stepTen = new JButton("Step 10 epochs");
        stepTen.addActionListener(event -> requestSteps(10));

        var reset = new JButton("Reset");
        reset.addActionListener(event -> requestReset());

        controls.add(pauseButton);
        controls.add(stepOne);
        controls.add(stepTen);
        controls.add(reset);
        return controls;
    }

    private void installMouseTracking() {
        addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseMoved(MouseEvent event) {
                if (!outputBounds.contains(event.getPoint())) {
                    return;
                }
                hoverX = clamp01((event.getX() - outputBounds.x) / (double) outputBounds.width);
                hoverY = clamp01(1.0 - (event.getY() - outputBounds.y) / (double) outputBounds.height);
                repaint();
            }
        });
    }

    private void startTraining() {
        trainingThread = Thread.ofPlatform()
                .name("jneuro-xor-training")
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

    private void updatePauseButton() {
        var button = pauseButton;
        if (button != null) {
            button.setText(paused ? "Resume" : "Pause");
        }
    }

    private void trainingLoop() {
        var history = new ArrayList<HistoryPoint>();
        var epoch = 0;
        var error = network.trainingError();

        try {
            while (running) {
                var epochs = nextTrainingBatch(error, epoch);
                if (!running) {
                    return;
                }

                if (consumeReset()) {
                    network = createNetwork();
                    epoch = 0;
                    error = network.trainingError();
                    history.clear();
                    publishSnapshot(epoch, error, history);
                    continue;
                }

                if (epochs == 0) {
                    synchronized (trainingLock) {
                        if (running && paused && pendingSteps == 0 && !resetRequested) {
                            trainingLock.wait();
                        }
                    }
                    continue;
                }

                for (int i = 0; i < epochs && epoch < MAX_EPOCHS && error > TARGET_ERROR; i++) {
                    error = network.trainEpoch();
                    epoch++;
                }
                publishSnapshot(epoch, error, history);

                if (error <= TARGET_ERROR || epoch >= MAX_EPOCHS) {
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

    private int nextTrainingBatch(double error, int epoch) {
        synchronized (trainingLock) {
            if (resetRequested) {
                return 0;
            }
            if (pendingSteps > 0) {
                var result = Math.min(pendingSteps, EPOCHS_PER_FRAME);
                pendingSteps -= result;
                return result;
            }
            if (!paused && error > TARGET_ERROR && epoch < MAX_EPOCHS) {
                return Math.min(EPOCHS_PER_FRAME, MAX_EPOCHS - epoch);
            }
            return 0;
        }
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

    private void publishSnapshot(int epoch, double error, List<HistoryPoint> history) {
        var image = NeuroXorGrid.render(network, OUTPUT_GRID_SIZE, gridInputs, gridOutputs);
        var bottom = (OUTPUT_GRID_SIZE - 1) * OUTPUT_GRID_SIZE;
        var output00 = gridOutputs[bottom];
        var output01 = gridOutputs[0];
        var output10 = gridOutputs[bottom + OUTPUT_GRID_SIZE - 1];
        var output11 = gridOutputs[OUTPUT_GRID_SIZE - 1];

        if (history.isEmpty() || history.get(history.size() - 1).epoch() != epoch) {
            history.add(new HistoryPoint(epoch, error, output00, output01, output10, output11));
        }

        var diagnostics = NeuroXorDiagnostics.capture(network, epoch, error);
        var hiddenImages = NeuroXorDiagnostics.renderHiddenMaps(diagnostics, HIDDEN_GRID_SIZE);
        snapshot = new FrameSnapshot(
                image,
                hiddenImages,
                diagnostics,
                history,
                error <= TARGET_ERROR);
        EventQueue.invokeLater(this::repaint);
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
            g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setColor(BACKGROUND);
            g.fillRect(0, 0, getWidth(), getHeight());

            var margin = 28;
            var mainTop = 52;
            var mainSize = Math.min(390, Math.max(280, getHeight() / 2 - 85));
            var outputLeft = 58;
            drawOutputSurface(g, current, outputLeft, mainTop, mainSize);

            var networkLeft = outputLeft + mainSize + 95;
            var networkWidth = Math.max(440, getWidth() - networkLeft - margin);
            drawNetwork(g, current, networkLeft, mainTop, networkWidth, mainSize);

            var hiddenTop = mainTop + mainSize + 70;
            var hiddenGap = 12;
            var hiddenSize = Math.min(
                    135,
                    Math.max(70, (getWidth() - 2 * margin - hiddenGap * 5) / 6));
            drawHiddenMaps(g, current, margin, hiddenTop, hiddenSize, hiddenGap);

            var historyTop = hiddenTop + hiddenSize + 66;
            var historyHeight = Math.max(120, getHeight() - historyTop - 28);
            drawHistory(g, current.history(), margin + 30, historyTop, getWidth() - 2 * margin - 30, historyHeight);
        } finally {
            g.dispose();
        }
    }

    private void drawOutputSurface(Graphics2D g, FrameSnapshot current, int left, int top, int size) {
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 17));
        g.setColor(FOREGROUND);
        g.drawString("Network output f(x, y)", left, top - 18);

        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(current.outputImage(), left, top, size, size, null);
        outputBounds.setBounds(left, top, size, size);

        g.setColor(FOREGROUND);
        g.drawRect(left, top, size, size);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 13));
        g.drawString("0", left - 4, top + size + 20);
        g.drawString("1", left + size - 4, top + size + 20);
        g.drawString("x", left + size / 2, top + size + 40);
        g.drawString("1", left - 21, top + 5);
        g.drawString("0", left - 21, top + size + 5);
        g.drawString("y", left - 38, top + size / 2);

        drawSample(g, left, top + size, 0, 7);
        drawSample(g, left, top, 1, 7);
        drawSample(g, left + size, top + size, 1, 7);
        drawSample(g, left + size, top, 0, 7);

        var probeX = left + (int) Math.round(hoverX * size);
        var probeY = top + (int) Math.round((1.0 - hoverY) * size);
        g.setColor(new Color(255, 80, 80));
        g.setStroke(new BasicStroke(1.5f));
        g.drawOval(probeX - 5, probeY - 5, 10, 10);
        g.drawLine(probeX - 9, probeY, probeX + 9, probeY);
        g.drawLine(probeX, probeY - 9, probeX, probeY + 9);

        var diagnostics = current.diagnostics();
        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
        g.setColor(FOREGROUND);
        g.drawString(String.format(
                Locale.ROOT,
                "epoch %,d    RMSE %.6f    %s",
                diagnostics.epoch(),
                diagnostics.error(),
                current.converged() ? "converged" : paused ? "paused" : "training"),
                left,
                top + size + 61);
        g.setColor(MUTED);
        g.drawString("Move the mouse over the surface to inspect a forward pass", left, top + size + 82);
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
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(images[neuron], x, top, size, size, null);
            g.setColor(MUTED);
            g.drawRect(x, top, size, size);

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

            drawSample(g, x, top + size, 0, 3);
            drawSample(g, x, top, 1, 3);
            drawSample(g, x + size, top + size, 1, 3);
            drawSample(g, x + size, top, 0, 3);

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

    private void drawHistory(Graphics2D g, List<HistoryPoint> history, int left, int top, int width, int height) {
        if (height <= 30 || history.isEmpty()) {
            return;
        }

        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
        g.setColor(FOREGROUND);
        g.drawString("Training history", left, top - 13);

        var chartTop = top + 8;
        var chartHeight = height - 35;
        var chartWidth = width - 10;
        g.setColor(new Color(50, 54, 62));
        g.fillRect(left, chartTop, chartWidth, chartHeight);
        g.setColor(MUTED);
        g.drawRect(left, chartTop, chartWidth, chartHeight);
        g.drawLine(left, chartTop + chartHeight / 2, left + chartWidth, chartTop + chartHeight / 2);

        var maxEpoch = Math.max(1, history.get(history.size() - 1).epoch());
        drawHistorySeries(g, history, left, chartTop, chartWidth, chartHeight, maxEpoch, RMSE_COLOR, 0);
        drawHistorySeries(g, history, left, chartTop, chartWidth, chartHeight, maxEpoch, OUTPUT_00_COLOR, 1);
        drawHistorySeries(g, history, left, chartTop, chartWidth, chartHeight, maxEpoch, OUTPUT_01_COLOR, 2);
        drawHistorySeries(g, history, left, chartTop, chartWidth, chartHeight, maxEpoch, OUTPUT_10_COLOR, 3);
        drawHistorySeries(g, history, left, chartTop, chartWidth, chartHeight, maxEpoch, OUTPUT_11_COLOR, 4);

        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));
        var legendX = left + 8;
        var legendY = chartTop + 15;
        legendItem(g, "RMSE", RMSE_COLOR, legendX, legendY);
        legendItem(g, "00", OUTPUT_00_COLOR, legendX + 70, legendY);
        legendItem(g, "01", OUTPUT_01_COLOR, legendX + 115, legendY);
        legendItem(g, "10", OUTPUT_10_COLOR, legendX + 160, legendY);
        legendItem(g, "11", OUTPUT_11_COLOR, legendX + 205, legendY);

        g.setColor(MUTED);
        g.drawString("1", left - 18, chartTop + 4);
        g.drawString("0", left - 18, chartTop + chartHeight + 4);
        g.drawString("epoch 0", left, chartTop + chartHeight + 18);
        g.drawString("epoch " + maxEpoch, left + chartWidth - 72, chartTop + chartHeight + 18);
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
                default -> throw new IllegalArgumentException("unknown series");
            };
            var y = top + (int) Math.round(height * (1.0 - clamp01(value)));
            if (previousX >= 0) {
                g.drawLine(previousX, previousY, x, y);
            }
            previousX = x;
            previousY = y;
        }
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

    private static void drawSample(Graphics2D g, int x, int y, int target, int radius) {
        g.setColor(target == 0 ? Color.BLACK : Color.WHITE);
        g.fillOval(x - radius, y - radius, radius * 2, radius * 2);
        g.setColor(BOUNDARY);
        g.drawOval(x - radius - 1, y - radius - 1, radius * 2 + 2, radius * 2 + 2);
    }

    private static void addXorSamples(Neuro network) {
        network.addTrainingSample(new double[]{0, 0}, new double[]{0});
        network.addTrainingSample(new double[]{0, 1}, new double[]{1});
        network.addTrainingSample(new double[]{1, 0}, new double[]{1});
        network.addTrainingSample(new double[]{1, 1}, new double[]{0});
    }

    private static double clamp01(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }
}

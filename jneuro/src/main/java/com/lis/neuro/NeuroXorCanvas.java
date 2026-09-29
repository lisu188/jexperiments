package com.lis.neuro;

import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.EventQueue;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.util.Locale;
import javax.swing.JFrame;
import javax.swing.WindowConstants;

public final class NeuroXorCanvas extends Canvas {
    private static final long serialVersionUID = 1L;
    private static final int GRID_SIZE = 256;
    private static final int EPOCHS_PER_FRAME = 10;
    private static final int MAX_EPOCHS = 10_000;
    private static final double TARGET_ERROR = 0.05;

    private final Neuro network;
    private final double[] gridInputs = NeuroXorGrid.createInputs(GRID_SIZE);
    private final double[] gridOutputs = new double[GRID_SIZE * GRID_SIZE];
    private volatile FrameSnapshot snapshot;
    private volatile boolean running = true;
    private Thread trainingThread;

    private record FrameSnapshot(
            BufferedImage image,
            int epoch,
            double error,
            double output00,
            double output01,
            double output10,
            double output11,
            boolean converged) {
    }

    public NeuroXorCanvas() {
        network = new Neuro(
                new int[]{2, 6, 1},
                Neuro.HyperParameters.defaults()
                        .withLearningRate(0.6)
                        .withMomentum(0.2)
                        .withSeed(42));
        addXorSamples(network);
        setPreferredSize(new Dimension(720, 820));
        setBackground(new Color(24, 24, 24));
        publishSnapshot(0, network.trainingError());
    }

    public static void main(String[] args) {
        EventQueue.invokeLater(() -> {
            var canvas = new NeuroXorCanvas();
            canvas.showWindow();
        });
    }

    private void showWindow() {
        var frame = new JFrame("JNeuro XOR training");
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        frame.add(this);
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

    private void startTraining() {
        trainingThread = Thread.ofPlatform()
                .name("jneuro-xor-training")
                .daemon()
                .start(this::trainingLoop);
    }

    private void stopTraining() {
        running = false;
        var thread = trainingThread;
        if (thread != null) {
            thread.interrupt();
        }
    }

    private void trainingLoop() {
        var epoch = 0;
        var error = network.trainingError();

        try {
            while (running && epoch < MAX_EPOCHS && error > TARGET_ERROR) {
                var end = Math.min(MAX_EPOCHS, epoch + EPOCHS_PER_FRAME);
                while (epoch < end && error > TARGET_ERROR) {
                    error = network.trainEpoch();
                    epoch++;
                }
                publishSnapshot(epoch, error);
                Thread.sleep(16);
            }

            if (running) {
                publishSnapshot(epoch, error);
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private void publishSnapshot(int epoch, double error) {
        var image = NeuroXorGrid.render(network, GRID_SIZE, gridInputs, gridOutputs);
        var bottom = (GRID_SIZE - 1) * GRID_SIZE;
        snapshot = new FrameSnapshot(
                image,
                epoch,
                error,
                gridOutputs[bottom],
                gridOutputs[0],
                gridOutputs[bottom + GRID_SIZE - 1],
                gridOutputs[GRID_SIZE - 1],
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

            var left = 70;
            var top = 32;
            var right = 30;
            var bottom = 150;
            var plotSize = Math.max(1, Math.min(getWidth() - left - right, getHeight() - top - bottom));

            g.setColor(getBackground());
            g.fillRect(0, 0, getWidth(), getHeight());

            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
            g.drawImage(current.image(), left, top, plotSize, plotSize, null);

            g.setColor(Color.LIGHT_GRAY);
            g.drawRect(left, top, plotSize, plotSize);
            g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 14));
            g.drawString("0", left - 5, top + plotSize + 22);
            g.drawString("1", left + plotSize - 4, top + plotSize + 22);
            g.drawString("x", left + plotSize / 2, top + plotSize + 44);
            g.drawString("1", left - 24, top + 5);
            g.drawString("0", left - 24, top + plotSize + 5);
            g.drawString("y", left - 42, top + plotSize / 2);

            drawSample(g, left, top + plotSize, 0);
            drawSample(g, left, top, 1);
            drawSample(g, left + plotSize, top + plotSize, 1);
            drawSample(g, left + plotSize, top, 0);

            var textY = top + plotSize + 72;
            g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 14));
            g.setColor(Color.LIGHT_GRAY);
            g.drawString(String.format(
                    Locale.ROOT,
                    "epoch %,d / %,d    RMSE %.6f    %s",
                    current.epoch(),
                    MAX_EPOCHS,
                    current.error(),
                    current.converged() ? "converged" : "training"),
                    left,
                    textY);
            g.drawString(String.format(
                    Locale.ROOT,
                    "00 %.4f    01 %.4f    10 %.4f    11 %.4f",
                    current.output00(),
                    current.output01(),
                    current.output10(),
                    current.output11()),
                    left,
                    textY + 28);
            g.drawString("pixel brightness = network(x, y)", left, textY + 56);
        } finally {
            g.dispose();
        }
    }

    private static void drawSample(Graphics2D g, int x, int y, int target) {
        var radius = 7;
        g.setColor(target == 0 ? Color.BLACK : Color.WHITE);
        g.fillOval(x - radius, y - radius, radius * 2, radius * 2);
        g.setColor(Color.RED);
        g.drawOval(x - radius - 1, y - radius - 1, radius * 2 + 2, radius * 2 + 2);
    }

    private static void addXorSamples(Neuro network) {
        network.addTrainingSample(new double[]{0, 0}, new double[]{0});
        network.addTrainingSample(new double[]{0, 1}, new double[]{1});
        network.addTrainingSample(new double[]{1, 0}, new double[]{1});
        network.addTrainingSample(new double[]{1, 1}, new double[]{0});
    }
}

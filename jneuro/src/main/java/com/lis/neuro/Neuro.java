package com.lis.neuro;

import java.util.ArrayList;
import java.util.List;

public class Neuro {
    private final int[] struct;
    private final int NW;
    private final double[][][] neuro;
    private final double[][][] prev;
    private final double alfa;
    private final double beta;
    private final double eta;
    private final List<Teacher> teachers = new ArrayList<>();
    private final List<Teacher> test = new ArrayList<>();
    public Neuro(int[] tab, double alfa, double beta, double eta) {
        this.alfa = alfa;
        this.beta = beta;
        this.eta = eta;
        struct = tab;
        NW = struct.length;
        neuro = new double[NW - 1][][];
        prev = new double[NW - 1][][];
        for (int i = 0; i < NW - 1; i++) {
            neuro[i] = new double[struct[i + 1]][];
            prev[i] = new double[struct[i + 1]][];
            for (int j = 0; j < struct[i + 1]; j++) {
                neuro[i][j] = new double[struct[i]];
                prev[i][j] = new double[struct[i]];
                for (int k = 0; k < struct[i]; k++) {
                    neuro[i][j][k] = prev[i][j][k] = Math.random() - 0.5;
                }
            }
        }
    }

    public Neuro(int[] tab) {
        this(tab, 0.2, 1.0, 0.8);
    }

    private double fcn(double x) {
        return 1 / (1 + Math.exp(-beta * x));
    }

    private double[] dfcn(double x[]) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = (1 - x[i]) * x[i];
        }
        return y;
    }

    private double[] useCut(double In[], int cut) {
        double Prev[] = In;
        double Out[] = In;
        for (int i = 0; i < struct.length - cut - 1; i++) {
            Out = new double[struct[i + 1]];
            for (int j = 0; j < struct[i + 1]; j++) {
                Out[j] = 0;
                for (int k = 0; k < struct[i]; k++) {
                    Out[j] += Prev[k] * neuro[i][j][k];
                }
                Out[j] = fcn(Out[j]);
            }
            Prev = Out;
        }
        return Out;
    }

    private double[][] O(int i) {
        double[][] O = new double[NW][];
        for (int j = NW - 1; j >= 0; j--) {
            O[NW - 1 - j] = useCut(teachers.get(i).In, j);
        }
        return O;
    }

    private double[][] E(int i) {
        double E[][];
        double O[][] = O(i);
        E = new double[NW - 1][];
        E[NW - 2] = new double[struct[NW - 1]];
        for (int j = 0; j < struct[NW - 1]; j++) {
            E[NW - 2][j] = teachers.get(i).Out[j] - O[NW - 1][j];
        }
        E[NW - 2] = dfcn(E[NW - 2]);

        for (int j = 3; NW - j >= 0; j++) {
            E[NW - j] = new double[struct[NW - j + 1]];
            for (int k = 0; k < struct[NW - j + 1]; k++) {
                E[NW - j][k] = 0;
            }
            for (int k = 0; k < struct[NW - j + 1]; k++) {
                for (int l = 0; l < struct[NW - j + 2]; l++) {
                    E[NW - j][k] += E[NW - j + 1][l] * neuro[NW - j + 1][l][k];
                }
            }
            for (int k = 0; k < struct[NW - j + 1]; k++) {
                E[NW - j][k] *= dfcn(O[NW - j + 1])[k];
            }
        }
        return E;
    }

    public void addTeacher(double In[], double Out[]) {
        if (struct[0] != In.length) {
            throw new RuntimeException("Wrong input array size");
        }
        if (struct[NW - 1] != Out.length) {
            throw new RuntimeException("Wrong output array size");
        }
        teachers.add(new Teacher(In, Out));
    }

    public void addTest(double In[], double Out[]) throws Exception {
        if (struct[0] != In.length) {
            throw new RuntimeException("Wrong input array size");
        }
        if (struct[NW - 1] != Out.length) {
            throw new RuntimeException("Wrong output array size");
        }
        test.add(new Teacher(In, Out));
    }

    public double[] use(double In[]) throws Exception {
        if (struct[0] != In.length) {
            throw new RuntimeException("Wrong input array size");
        }
        double Prev[] = In;
        double Out[] = null;
        for (int i = 0; i < struct.length - 1; i++) {
            Out = new double[struct[i + 1]];
            for (int j = 0; j < struct[i + 1]; j++) {
                Out[j] = 0;
                for (int k = 0; k < struct[i]; k++) {
                    Out[j] += Prev[k] * neuro[i][j][k];
                }
                Out[j] = fcn(Out[j]);
            }
            Prev = Out;
        }
        return Out;
    }

    public void teach(int ite) throws Exception {
        if (teachers.size() != 0) {
            double[][] E;
            double[][] O;
            while (ite-- > 0) {
                double[][][] diff = new double[NW - 1][][];
                for (int i = 0; i < NW - 1; i++) {
                    diff[i] = new double[struct[i + 1]][];
                    for (int j = 0; j < struct[i + 1]; j++) {
                        diff[i][j] = new double[struct[i]];
                        for (int k = 0; k < struct[i]; k++) {
                            diff[i][j][k] = neuro[i][j][k] - prev[i][j][k];
                            prev[i][j][k] = neuro[i][j][k];
                        }
                    }
                }

                for (int i = 0; i < teachers.size(); i++) {
                    O = O(i);
                    E = E(i);

                    for (int j = 0; j < NW - 1; j++) {
                        for (int k = 0; k < struct[j + 1]; k++) {
                            for (int l = 0; l < struct[j]; l++) {
                                neuro[j][k][l] += eta * E[j][k] * O[j][l];
                            }
                        }
                    }
                }
                for (int i = 0; i < NW - 1; i++) {
                    for (int j = 0; j < struct[i + 1]; j++) {
                        for (int k = 0; k < struct[i]; k++) {
                            neuro[i][j][k] += alfa * diff[i][j][k];
                        }
                    }
                }
            }
        }
    }

    public void teach(double ERMS) throws Exception {
        while (ERMS() > ERMS) {
            teach(1);
        }
    }

    public double ERMS() {
        double ERMS = 0;
        double tmp;
        double[] OutT, OutN;
        int size = teachers.size();
        for (Teacher teacher : teachers) {
            tmp = 0;
            OutT = teacher.Out;
            OutN = useCut(teacher.In, 0);
            for (int k = 0; k < struct[NW - 1]; k++) {
                tmp += Math.pow(OutT[k] - OutN[k], 2);
            }
            ERMS += Math.sqrt(tmp / struct[NW - 1]);
        }
        ERMS = ERMS / size;
        return ERMS;
    }

    public double TEST() {
        double ERMS = 0;
        double tmp;
        double[] OutT, OutN;
        int size = test.size();
        for (Teacher aTest : test) {
            tmp = 0;
            OutT = aTest.Out;
            OutN = useCut(aTest.In, 0);
            for (int k = 0; k < struct[NW - 1]; k++) {
                tmp += Math.pow(OutT[k] - OutN[k], 2);
            }
            ERMS += Math.sqrt(tmp / struct[NW - 1]);
        }
        return ERMS / (double) size;
    }

    private class Teacher {
        private final double[] In;
        private final double[] Out;

        private Teacher(double[] X, double[] Y) {
            In = X;
            Out = Y;
        }
    }
}

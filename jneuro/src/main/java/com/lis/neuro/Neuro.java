package com.lis.neuro;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

class Neuro {
    private final int[] _str;

    private final double _alfa;
    private final double _beta;
    private final double _eta;
    private final int _nw;
    private final List<Pair<double[], double[]>> teachers = new ArrayList<>();
    private final List<Pair<double[], double[]>> tests = new ArrayList<>();
    private final double[][][] _neuro;
    private final double[][][] _prev;
    private final double[][][] _diff;
    private final double[][] _o;
    private final double[][] _e;

    private Neuro(int[] str,
                  double alfa,
                  double beta,
                  double eta
    ) {
        this._str = str;
        this._alfa = alfa;
        this._beta = beta;
        this._eta = eta;
        this._nw = str.length;
        _neuro = new double[_nw - 1][][];
        _prev = new double[_nw - 1][][];
        _diff = new double[_nw - 1][][];
        for (int i = 0; i < _nw - 1; i++) {
            _neuro[i] = new double[_str[i + 1]][];
            _prev[i] = new double[_str[i + 1]][];
            _diff[i] = new double[_str[i + 1]][];
            for (int j = 0; j < _str[i + 1]; j++) {
                _neuro[i][j] = new double[_str[i]];
                _prev[i][j] = new double[_str[i]];
                _diff[i][j] = new double[_str[i]];
                for (int k = 0; k < _str[i]; k++) {
                    _diff[i][j][k] = _neuro[i][j][k] = _prev[i][j][k] = Math.random() - 0.5;
                }
            }
        }

        _e = new double[_nw - 1][];
        for (int i = 0; i < _nw - 1; i++) {
            _e[i] = new double[_str[i + 1]];
        }

        _o = new double[_nw][];
        for (int i = 0; i < _nw; i++) {
            _o[i] = new double[_str[i]];
        }
    }

    public static void main(String... args) {
        Neuro neuro = new Neuro(new int[]{2, 5, 25, 5, 1}, 0.2, 1, 0.8);
        neuro.add_teacher(new double[]{1, 1}, new double[]{1});
        neuro.add_teacher(new double[]{1, 0}, new double[]{1});
        neuro.add_teacher(new double[]{0, 1}, new double[]{1});
        neuro.add_teacher(new double[]{0, 0}, new double[]{0});
        System.out.println(neuro.teach(0.001));

    }

    private double fcn(double x, double beta) {
        return 1.0 / (1.0 + Math.exp(-beta * x));
    }

    private double dfcn(double x) {
        return (1.0 - x) * x;
    }

    private void o(double[] t) {
        System.arraycopy(t, 0, _o[0], 0, _str[0]);
        for (int i = 1; i < _nw; i++) {
            for (int j = 0; j < _str[i]; j++) {
                _o[i][j] = 0;
                for (int k = 0; k < _str[i - 1]; k++) {
                    _o[i][j] += _o[i - 1][k] * _neuro[i - 1][j][k];
                }
                _o[i][j] = fcn(_o[i][j], _beta);
            }
        }
    }

    private void e(double[] in, double[] out) {
        o(in);
        for (int i = 0; i < _str[_nw - 1]; i++) {
            _e[_nw - 2][i] = dfcn(out[i] - _o[_nw - 1][i]);
        }

        for (int j = 3; _nw - j >= 0; j++) {
            for (int k = 0; k < _str[_nw - j + 1]; k++) {
                _e[_nw - j][k] = 0;
            }
            for (int k = 0; k < _str[_nw - j + 1]; k++) {
                for (int l = 0; l < _str[_nw - j + 2]; l++) {
                    _e[_nw - j][k] += _e[_nw - j + 1][l] * _neuro[_nw - j + 1][l][k];
                }
            }
            for (int k = 0; k < _str[_nw - j + 1]; k++) {
                _e[_nw - j][k] *= dfcn(_o[_nw - j + 1][k]);
            }
        }
    }

    private void add_teacher(double[] in, double[] out) {
        teachers.add(new Pair<>(in, out));
    }

    void add_test(double[] in, double[] out) {
        tests.add(new Pair<>(in, out));
    }

    private int teach(double erms, int step) {
        int i = 0;
        while (this.erms() > erms) {
            teach(step);
            i += step;
        }
        return i;
    }

    private int teach(double erms) {
        return teach(erms, 1);
    }

    private void teach(int ite) {
        if (teachers.size() != 0) {
            while (ite-- > 0) {
                for (int i = 0; i < _nw - 1; i++) {
                    for (int j = 0; j < _str[i + 1]; j++) {
                        for (int k = 0; k < _str[i]; k++) {
                            _diff[i][j][k] = _neuro[i][j][k] - _prev[i][j][k];
                            _prev[i][j][k] = _neuro[i][j][k];
                        }
                    }
                }

                Collections.shuffle(teachers);

                for (Pair<double[], double[]> t : teachers) {
                    o(t.first);
                    e(t.first, t.second);

                    for (int j = 0; j < _nw - 1; j++) {
                        for (int k = 0; k < _str[j + 1]; k++) {
                            for (int l = 0; l < _str[j]; l++) {
                                _neuro[j][k][l] += (_eta * _e[j][k] * _o[j][l]) / teachers.size();
                            }
                        }
                    }
                }

                for (int i = 0; i < _nw - 1; i++) {
                    for (int j = 0; j < _str[i + 1]; j++) {
                        for (int k = 0; k < _str[i]; k++) {
                            _neuro[i][j][k] += _alfa * _diff[i][j][k];
                        }
                    }
                }
            }
        }
    }

    private double erms() {
        double erms = 0;
        double tmp;
        double[] out_t;
        for (Pair<double[], double[]> teacher : teachers) {
            tmp = 0;
            out_t = teacher.second;
            o(teacher.first);
            for (int k = 0; k < _str[_nw - 1]; k++) {
                tmp += Math.pow(out_t[k] - _o[_nw - 1][k], 2);
            }
            erms += Math.sqrt(tmp / _str[_nw - 1]);
        }
        return erms / teachers.size();
    }

    double test() {
        double erms = 0;
        double tmp;
        double[] out_t;
        for (Pair<double[], double[]> teacher : tests) {
            tmp = 0;
            out_t = teacher.second;
            o(teacher.first);
            for (int k = 0; k < _str[_nw - 1]; k++) {
                tmp += Math.pow(out_t[k] - _o[_nw - 1][k], 2);
            }
            erms += Math.sqrt(tmp / _str[_nw - 1]);
        }
        return erms / teachers.size();
    }

    private static final class Pair<T, U> {
        final T first;
        final U second;

        private Pair(T first, U second) {
            this.first = first;
            this.second = second;
        }
    }
}
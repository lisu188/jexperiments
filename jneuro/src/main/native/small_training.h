#pragma once
#include <cmath>
#include <cstddef>

// Experimental fixed topology 2 -> 8 -> 8 -> 8 -> 1, FP64 only.
// State is SoA: (177 parameters + 177 velocities) x model count.
constexpr int parameter_count = 177;
constexpr int widths[] = {2, 8, 8, 8, 1};
constexpr int parameter_offsets[] = {0, 24, 96, 168};
constexpr int activation_offsets[] = {0, 2, 10, 18, 26};
constexpr int delta_offsets[] = {0, 8, 16, 24};

struct SmallArguments {
    double* state;
    const double* inputs;
    const double* targets;
    const int* orders;
    int models, samples, epochs, batch, online;
    double learning_rate, momentum, beta;
    int sigmoid;
};

struct Scalar {
    using Value = double;
    static constexpr int lanes = 1;
    static Value set(double a) { return a; }
    static Value load(const double* a) { return *a; }
    static void store(double* a, Value b) { *a = b; }
    static Value add(Value a, Value b) { return a + b; }
    static Value sub(Value a, Value b) { return a - b; }
    static Value mul(Value a, Value b) { return a * b; }
    static Value fma(Value a, Value b, Value c) { return std::fma(a, b, c); }
    static Value activate(Value value, int mode) {
        auto negative_exp = [](double input) {
            if (std::isnan(input)) return input;
            if (input <= -745.0) return 0.0;
            const int exponent = static_cast<int>(input * 1.4426950408889634);
            const double remainder = input - exponent * 0.6931471805599453;
            const double square = remainder * remainder;
            return std::ldexp(1.0 + remainder + square * (0.5 + remainder * (0.16666666666666666 +
                remainder * (0.041666666666666664 + remainder * 0.008333333333333333))), exponent);
        };
        if (!mode) return 1.0 / (1.0 + std::exp(-value));
        if (value >= 0.0) return 1.0 / (1.0 + negative_exp(-value));
        const double exponential = negative_exp(value);
        return exponential / (1.0 + exponential);
    }
};

template <class Ops> void train_group(const SmallArguments& args, int first_model) {
    using V = typename Ops::Value;
    V state[parameter_count * 2], activation[27], delta[25], gradient[parameter_count];
    for (int index = 0; index < parameter_count * 2; ++index)
        state[index] = Ops::load(args.state + static_cast<std::size_t>(index) * args.models + first_model);
    const V beta = Ops::set(args.beta), momentum = Ops::set(args.momentum), rate = Ops::set(args.learning_rate);
    const V one = Ops::set(1), zero = Ops::set(0);
    for (int epoch = 0; epoch < args.epochs; ++epoch) {
        for (int start = 0; start < args.samples; start += args.batch) {
            const int count = args.batch < args.samples - start ? args.batch : args.samples - start;
            for (int index = 0; index < parameter_count; ++index) gradient[index] = zero;
            for (int position = 0; position < count; ++position) {
                double x[Ops::lanes], y[Ops::lanes], target[Ops::lanes];
                for (int lane = 0; lane < Ops::lanes; ++lane) {
                    const int sample = args.orders[(static_cast<std::size_t>(first_model + lane) * args.epochs + epoch) * args.samples + start + position];
                    x[lane] = args.inputs[sample * 2]; y[lane] = args.inputs[sample * 2 + 1]; target[lane] = args.targets[sample];
                }
                activation[0] = Ops::load(x); activation[1] = Ops::load(y);
                for (int layer = 0; layer < 4; ++layer) {
                    const int in = widths[layer], out = widths[layer + 1], offset = parameter_offsets[layer];
                    for (int neuron = 0; neuron < out; ++neuron) {
                        V sum = state[offset + in * out + neuron];
                        for (int input = 0; input < in; ++input)
                            sum = Ops::fma(activation[activation_offsets[layer] + input], state[offset + neuron * in + input], sum);
                        activation[activation_offsets[layer + 1] + neuron] = Ops::activate(Ops::mul(sum, beta), args.sigmoid);
                    }
                }
                const V output = activation[26];
                delta[24] = Ops::mul(Ops::mul(Ops::mul(Ops::sub(Ops::load(target), output), beta), output), Ops::sub(one, output));
                for (int layer = 2; layer >= 0; --layer) {
                    for (int neuron = 0; neuron < 8; ++neuron) {
                        V sum = args.online ? Ops::mul(delta[delta_offsets[layer + 1]], state[parameter_offsets[layer + 1] + neuron]) : zero;
                        for (int next = args.online ? 1 : 0; next < widths[layer + 2]; ++next)
                            sum = Ops::fma(delta[delta_offsets[layer + 1] + next], state[parameter_offsets[layer + 1] + next * 8 + neuron], sum);
                        const V value = activation[activation_offsets[layer + 1] + neuron];
                        delta[delta_offsets[layer] + neuron] = args.online
                            ? Ops::mul(sum, Ops::mul(Ops::mul(beta, value), Ops::sub(one, value)))
                            : Ops::mul(Ops::mul(Ops::mul(sum, beta), value), Ops::sub(one, value));
                    }
                }
                for (int layer = 0; layer < 4; ++layer) {
                    const int in = widths[layer], out = widths[layer + 1], offset = parameter_offsets[layer];
                    for (int neuron = 0; neuron < out; ++neuron) {
                        const V change = delta[delta_offsets[layer] + neuron];
                        for (int input = 0; input < in; ++input) {
                            const int index = offset + neuron * in + input;
                            const V source = activation[activation_offsets[layer] + input];
                            gradient[index] = args.online ? Ops::mul(Ops::mul(rate, change), source) : Ops::fma(source, change, gradient[index]);
                        }
                        const int index = offset + in * out + neuron;
                        gradient[index] = args.online ? Ops::mul(rate, change) : Ops::add(gradient[index], change);
                    }
                }
            }
            const V scale = Ops::set(args.learning_rate / count);
            for (int index = 0; index < parameter_count; ++index) {
                const V change = args.online ? gradient[index] : Ops::mul(scale, gradient[index]);
                state[parameter_count + index] = Ops::fma(momentum, state[parameter_count + index], change);
                state[index] = Ops::add(state[index], state[parameter_count + index]);
            }
        }
    }
    for (int index = 0; index < parameter_count * 2; ++index)
        Ops::store(args.state + static_cast<std::size_t>(index) * args.models + first_model, state[index]);
}

void train_avx2(const SmallArguments& args, int first_model);

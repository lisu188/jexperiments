// CUDA ABI 1. Packed activations are sample-major; weights are output-major.
// Every reduction has one owning thread and a fixed summation order. Compile
// without implicit FMA contraction: the explicit fma calls match Neuro's scalar
// arithmetic. All kernels use a one-dimensional grid and 64-bit array offsets.
#include <cmath>
#include <cstddef>

namespace {
__device__ double fast_exp_negative(double value) {
    if (value <= -745.0) return 0.0;
    const int exponent = static_cast<int>(value * 1.4426950408889634);
    const double remainder = value - exponent * 0.6931471805599453;
    const double square = remainder * remainder;
    const double polynomial = 1.0 + remainder + square * (0.5 + remainder *
        (0.16666666666666666 + remainder *
        (0.041666666666666664 + remainder * 0.008333333333333333)));
    return ldexp(polynomial, exponent);
}

__device__ double activate(double value, int sigmoid_mode) {
    if (sigmoid_mode == 0) return 1.0 / (1.0 + exp(-value));
    if (value >= 0.0) return 1.0 / (1.0 + fast_exp_negative(-value));
    const double exponential = fast_exp_negative(value);
    return exponential / (1.0 + exponential);
}

__device__ std::size_t thread_index() {
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}
}  // namespace

extern "C" __global__ void forward(
    const double* source, const double* weights, const double* biases,
    double* output, int inputs, int outputs, int count, double beta,
    int sigmoid_mode) {
    const std::size_t index = thread_index();
    if (index >= static_cast<std::size_t>(count) * outputs) return;
    const std::size_t source_offset = index / outputs * inputs;
    const int neuron = static_cast<int>(index % outputs);
    const std::size_t weight_offset = static_cast<std::size_t>(neuron) * inputs;
    double sum = biases[neuron];
    for (int input = 0; input < inputs; ++input) {
        sum = fma(source[source_offset + input], weights[weight_offset + input], sum);
    }
    output[index] = activate(sum * beta, sigmoid_mode);
}

extern "C" __global__ void output_delta(
    const double* activations, const double* targets, double* delta,
    int outputs, int count, double beta) {
    const std::size_t index = thread_index();
    if (index >= static_cast<std::size_t>(count) * outputs) return;
    const double activation = activations[index];
    delta[index] = (targets[index] - activation) * beta * activation * (1.0 - activation);
}

extern "C" __global__ void hidden_delta(
    const double* next_weights, const double* next_delta,
    const double* activations, double* delta, int current_width,
    int next_width, int count, double beta) {
    const std::size_t index = thread_index();
    if (index >= static_cast<std::size_t>(count) * current_width) return;
    const std::size_t next_offset = index / current_width * next_width;
    const int neuron = static_cast<int>(index % current_width);
    double sum = next_delta[next_offset] * next_weights[neuron];
    for (int next = 1; next < next_width; ++next) {
        sum = fma(next_delta[next_offset + next],
            next_weights[static_cast<std::size_t>(next) * current_width + neuron], sum);
    }
    const double activation = activations[index];
    delta[index] = sum * (beta * activation * (1.0 - activation));
}

// Launch inputs*outputs + outputs threads: one owner per weight or bias.
// online must be 0 or 1; online=1 requires count=1. Host validates dimensions.
extern "C" __global__ void update(
    const double* source, const double* delta, double* weights, double* biases,
    double* weight_velocity, double* bias_velocity, int inputs, int outputs,
    int count, double learning_rate, double momentum, int online) {
    const std::size_t index = thread_index();
    const std::size_t weight_count = static_cast<std::size_t>(inputs) * outputs;
    if (index >= weight_count + outputs) return;
    if (index < weight_count) {
        const int neuron = static_cast<int>(index / inputs);
        const int input = static_cast<int>(index % inputs);
        double scaled_gradient;
        if (online) {
            const double scale = learning_rate * delta[neuron];
            scaled_gradient = scale * source[input];
        } else {
            double gradient = 0.0;
            for (int sample = 0; sample < count; ++sample) {
                gradient = fma(source[static_cast<std::size_t>(sample) * inputs + input],
                    delta[static_cast<std::size_t>(sample) * outputs + neuron], gradient);
            }
            scaled_gradient = (learning_rate / count) * gradient;
        }
        const double velocity = fma(momentum, weight_velocity[index], scaled_gradient);
        weight_velocity[index] = velocity;
        weights[index] += velocity;
    } else {
        const int neuron = static_cast<int>(index - weight_count);
        double scaled_gradient;
        if (online) {
            scaled_gradient = learning_rate * delta[neuron];
        } else {
            double gradient = 0.0;
            for (int sample = 0; sample < count; ++sample) {
                gradient += delta[static_cast<std::size_t>(sample) * outputs + neuron];
            }
            scaled_gradient = (learning_rate / count) * gradient;
        }
        const double velocity = fma(momentum, bias_velocity[neuron], scaled_gradient);
        bias_velocity[neuron] = velocity;
        biases[neuron] += velocity;
    }
}

// Launch count*max(inputs, outputs) threads. Order is the JVM-generated shuffle.
extern "C" __global__ void gather(
    const double* all_inputs, const double* all_targets, const int* order,
    double* batch_inputs, double* batch_targets, int inputs, int outputs,
    int start, int count) {
    const std::size_t index = thread_index();
    if (index < static_cast<std::size_t>(count) * inputs) {
        const std::size_t sample = static_cast<std::size_t>(order[start + index / inputs]);
        batch_inputs[index] = all_inputs[sample * inputs + index % inputs];
    }
    if (index < static_cast<std::size_t>(count) * outputs) {
        const std::size_t sample = static_cast<std::size_t>(order[start + index / outputs]);
        batch_targets[index] = all_targets[sample * outputs + index % outputs];
    }
}

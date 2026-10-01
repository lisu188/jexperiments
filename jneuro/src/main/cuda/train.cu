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

// SMALL ABI 2: one cooperative block owns one complete tiny network. Host
// eligibility is 2 inputs, 1..4 hidden layers of 4/8/16, and one output.
// A tile of eight samples bounds FP64 shared storage below 32 KiB. Gradient
// owners visit samples in exactly the reference order, including across tiles.
namespace {
constexpr int SMALL_PARAMETERS = 881;
constexpr int SMALL_ACTIVATIONS = 67;
constexpr int SMALL_DELTAS = 65;
constexpr int SMALL_TILE = 8;

template <typename Real> __device__ Real small_fma(Real a, Real b, Real c);
template <> __device__ double small_fma(double a, double b, double c) { return fma(a, b, c); }
template <> __device__ float small_fma(float a, float b, float c) { return fmaf(a, b, c); }
template <typename Real> __device__ Real small_exp(Real value);
template <> __device__ double small_exp(double value) { return exp(value); }
template <> __device__ float small_exp(float value) { return expf(value); }
template <typename Real> __device__ Real small_ldexp(Real value, int exponent);
template <> __device__ double small_ldexp(double value, int exponent) { return ldexp(value, exponent); }
template <> __device__ float small_ldexp(float value, int exponent) { return ldexpf(value, exponent); }

template <typename Real> __device__ Real small_exp_negative(Real value) {
    if (value <= Real(sizeof(Real) == sizeof(double) ? -745.0 : -103.0)) return Real(0);
    const int exponent = static_cast<int>(value * Real(1.4426950408889634));
    const Real remainder = value - Real(exponent) * Real(0.6931471805599453);
    const Real square = remainder * remainder;
    const Real polynomial = Real(1) + remainder + square * (Real(0.5) + remainder *
        (Real(0.16666666666666666) + remainder *
        (Real(0.041666666666666664) + remainder * Real(0.008333333333333333))));
    return small_ldexp(polynomial, exponent);
}

template <typename Real> __device__ Real small_activate(Real value, int mode) {
    if (mode == 0) return Real(1) / (Real(1) + small_exp(-value));
    if (value >= Real(0)) return Real(1) / (Real(1) + small_exp_negative(-value));
    const Real exponential = small_exp_negative(value);
    return exponential / (Real(1) + exponential);
}

template <typename Real, typename Storage = double, int Parameters = SMALL_PARAMETERS,
          int Activations = SMALL_ACTIVATIONS, int Deltas = SMALL_DELTAS>
__device__ void small_train(
    Storage* packed, const Storage* inputs, const Storage* targets, const int* orders,
    const int* topology, const int* active, int models, int layers, int samples,
    int epochs, int batch_size, double learning_rate_double, double momentum_double,
    double beta_double, int mode, int online, const int* requests = nullptr, Storage* checkpoint = nullptr) {
    const int model = blockIdx.x;
    if (model >= models || (requests == nullptr && !active[model])) return;
    // Search ABI 3: [state offset, topology offset, order offset, layers,
    // epochs, checkpoint offset] per block; every offset counts array elements.
    if (requests != nullptr) {
        const int* request = requests + model * 6;
        packed += request[0];
        topology += request[1];
        orders += request[2];
        layers = request[3];
        epochs = request[4];
        checkpoint += request[5];
    }
    const int thread = threadIdx.x;
    __shared__ Real state[2 * Parameters];
    __shared__ Real activation[SMALL_TILE * Activations];
    __shared__ Real delta[SMALL_TILE * Deltas];
    __shared__ Real gradient[Parameters];
    __shared__ int invalid;
    // Block-uniform metadata belongs to shared storage, not a per-thread
    // dynamically indexed array that spills to CUDA local memory.
    __shared__ int parameter_offset[5], activation_offset[6], delta_offset[5];
    __shared__ int parameters, activation_count, delta_count;
    if (thread == 0) {
      parameters = 0; activation_count = topology[0]; delta_count = 0;
      activation_offset[0] = 0;
      for (int layer = 0; layer < layers; ++layer) {
        parameter_offset[layer] = parameters;
        parameters += topology[layer + 1] * (topology[layer] + 1);
        activation_offset[layer + 1] = activation_count;
        activation_count += topology[layer + 1];
        delta_offset[layer] = delta_count;
        delta_count += topology[layer + 1];
      }
    }
    __syncthreads();
    const std::size_t base = requests == nullptr ? static_cast<std::size_t>(model) * parameters * 2 : 0;
    const std::size_t order_base = requests == nullptr ? static_cast<std::size_t>(model) * epochs * samples : 0;
    for (int index = thread; index < parameters * 2; index += blockDim.x)
        state[index] = Real(packed[base + index]);
    if (thread == 0) invalid = 0;
    __syncthreads();
    const Real learning_rate = Real(learning_rate_double);
    const Real momentum = Real(momentum_double);
    const Real beta = Real(beta_double);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int start = 0; start < samples; start += batch_size) {
            const int count = min(batch_size, samples - start);
            for (int index = thread; index < parameters; index += blockDim.x) gradient[index] = Real(0);
            __syncthreads();
            for (int tile = 0; tile < count; tile += SMALL_TILE) {
                const int tile_count = min(SMALL_TILE, count - tile);
                for (int index = thread; index < tile_count * topology[0]; index += blockDim.x) {
                    const int row = index / topology[0];
                    const int sample = orders[order_base + static_cast<std::size_t>(epoch) * samples + start + tile + row];
                    activation[row * activation_count + index % topology[0]] = Real(inputs[static_cast<std::size_t>(sample) * topology[0] + index % topology[0]]);
                }
                __syncthreads();
                for (int layer = 0; layer < layers; ++layer) {
                    const int in = topology[layer], out = topology[layer + 1];
                    for (int index = thread; index < tile_count * out; index += blockDim.x) {
                        const int row = index / out, neuron = index % out;
                        Real sum = state[parameter_offset[layer] + in * out + neuron];
                        for (int input = 0; input < in; ++input)
                            sum = small_fma(activation[row * activation_count + activation_offset[layer] + input],
                                state[parameter_offset[layer] + neuron * in + input], sum);
                        activation[row * activation_count + activation_offset[layer + 1] + neuron] = small_activate(sum * beta, mode);
                    }
                    __syncthreads();
                }
                for (int row = thread; row < tile_count; row += blockDim.x) {
                    const int sample = orders[order_base + static_cast<std::size_t>(epoch) * samples + start + tile + row];
                    const Real value = activation[row * activation_count + activation_offset[layers]];
                    delta[row * delta_count + delta_offset[layers - 1]] = (Real(targets[sample]) - value) * beta * value * (Real(1) - value);
                }
                __syncthreads();
                for (int layer = layers - 2; layer >= 0; --layer) {
                    const int width = topology[layer + 1], next_width = topology[layer + 2];
                    for (int index = thread; index < tile_count * width; index += blockDim.x) {
                        const int row = index / width, neuron = index % width;
                        Real sum = online ? delta[row * delta_count + delta_offset[layer + 1]] * state[parameter_offset[layer + 1] + neuron] : Real(0);
                        for (int next = online ? 1 : 0; next < next_width; ++next)
                            sum = small_fma(delta[row * delta_count + delta_offset[layer + 1] + next],
                                state[parameter_offset[layer + 1] + next * width + neuron], sum);
                        const Real value = activation[row * activation_count + activation_offset[layer + 1] + neuron];
                        delta[row * delta_count + delta_offset[layer] + neuron] = online
                            ? sum * (beta * value * (Real(1) - value)) : sum * beta * value * (Real(1) - value);
                    }
                    __syncthreads();
                }
                for (int layer = 0; layer < layers; ++layer) {
                    const int in = topology[layer], out = topology[layer + 1];
                    for (int index = thread; index < in * out + out; index += blockDim.x) {
                        const bool weight = index < in * out;
                        const int neuron = weight ? index / in : index - in * out;
                        Real sum = gradient[parameter_offset[layer] + index];
                        for (int row = 0; row < tile_count; ++row) {
                            const Real change = delta[row * delta_count + delta_offset[layer] + neuron];
                            if (online) sum = weight ? (learning_rate * change) * activation[row * activation_count + activation_offset[layer] + index % in] : learning_rate * change;
                            else if (weight) sum = small_fma(activation[row * activation_count + activation_offset[layer] + index % in], change, sum);
                            else sum += change;
                        }
                        gradient[parameter_offset[layer] + index] = sum;
                    }
                }
                __syncthreads();
            }
            for (int index = thread; index < parameters; index += blockDim.x) {
                const Real scaled = online ? gradient[index] : (learning_rate / Real(count)) * gradient[index];
                const Real velocity = small_fma(momentum, state[parameters + index], scaled);
                state[parameters + index] = velocity;
                state[index] += velocity;
                if (!isfinite(state[index]) || !isfinite(velocity)) atomicExch(&invalid, 1);
            }
            __syncthreads();
            if (invalid) break;
        }
        if (invalid) break;
    }
    if (checkpoint != nullptr) {
        // Numerical failure belongs to this lane. Keep its resident checkpoint
        // intact and let unrelated models complete; driver errors remain fatal.
        if (thread == 0) checkpoint[0] = invalid ? 1.0 : 0.0;
        for (int index = thread; index < parameters * 2; index += blockDim.x) {
            checkpoint[index + 1] = invalid ? packed[base + index] : Storage(state[index]);
            if (!invalid) packed[base + index] = Storage(state[index]);
        }
    } else {
        for (int index = thread; index < parameters * 2; index += blockDim.x)
            packed[base + index] = invalid ? Storage(NAN) : Storage(state[index]);
    }
}
} // namespace

extern "C" __global__ void small_train_fp64(double* packed, const double* inputs, const double* targets,
    const int* orders, const int* topology, const int* active, int models, int layers, int samples,
    int epochs, int batch_size, double learning_rate, double momentum, double beta, int mode, int online) {
    small_train<double>(packed, inputs, targets, orders, topology, active, models, layers, samples,
        epochs, batch_size, learning_rate, momentum, beta, mode, online);
}
extern "C" __global__ void small_train_fp32(double* packed, const double* inputs, const double* targets,
    const int* orders, const int* topology, const int* active, int models, int layers, int samples,
    int epochs, int batch_size, double learning_rate, double momentum, double beta, int mode, int online) {
    small_train<float>(packed, inputs, targets, orders, topology, active, models, layers, samples,
        epochs, batch_size, learning_rate, momentum, beta, mode, online);
}


// Heterogeneous search batches share immutable data, but every block has its
// own topology, optimizer state, shuffle order and bounded epoch count.
extern "C" __global__ void search_train_fp64(double* packed, const double* inputs, const double* targets,
    const int* orders, const int* topology, const int* requests, double* checkpoint,
    int models, int samples, int batch_size, double learning_rate, double momentum, double beta, int mode, int online) {
    small_train<double>(packed, inputs, targets, orders, topology, nullptr, models, 0, samples,
        0, batch_size, learning_rate, momentum, beta, mode, online, requests, checkpoint);
}
extern "C" __global__ void search_train_fp32(double* packed, const double* inputs, const double* targets,
    const int* orders, const int* topology, const int* requests, double* checkpoint,
    int models, int samples, int batch_size, double learning_rate, double momentum, double beta, int mode, int online) {
    small_train<float>(packed, inputs, targets, orders, topology, nullptr, models, 0, samples,
        0, batch_size, learning_rate, momentum, beta, mode, online, requests, checkpoint);
}

// General FP32 ABI: all tensors and floating scalar arguments are true float.
// FP64 entrypoints above retain their original operation order.
extern "C" __global__ void forward_fp32(
    const float* source, const float* weights, const float* biases,
    float* output, int inputs, int outputs, int count, float beta,
    int sigmoid_mode) {
    const std::size_t index = thread_index();
    if (index >= static_cast<std::size_t>(count) * outputs) return;
    const std::size_t source_offset = index / outputs * inputs;
    const int neuron = static_cast<int>(index % outputs);
    const std::size_t weight_offset = static_cast<std::size_t>(neuron) * inputs;
    float sum = biases[neuron];
    for (int input = 0; input < inputs; ++input) {
        sum = fmaf(source[source_offset + input], weights[weight_offset + input], sum);
    }
    output[index] = small_activate<float>(sum * beta, sigmoid_mode);
}

extern "C" __global__ void output_delta_fp32(
    const float* activations, const float* targets, float* delta,
    int outputs, int count, float beta) {
    const std::size_t index = thread_index();
    if (index >= static_cast<std::size_t>(count) * outputs) return;
    const float activation = activations[index];
    delta[index] = (targets[index] - activation) * beta * activation * (1.0f - activation);
}

extern "C" __global__ void hidden_delta_fp32(
    const float* next_weights, const float* next_delta,
    const float* activations, float* delta, int current_width,
    int next_width, int count, float beta) {
    const std::size_t index = thread_index();
    if (index >= static_cast<std::size_t>(count) * current_width) return;
    const std::size_t next_offset = index / current_width * next_width;
    const int neuron = static_cast<int>(index % current_width);
    float sum = next_delta[next_offset] * next_weights[neuron];
    for (int next = 1; next < next_width; ++next) {
        sum = fmaf(next_delta[next_offset + next],
            next_weights[static_cast<std::size_t>(next) * current_width + neuron], sum);
    }
    const float activation = activations[index];
    delta[index] = sum * (beta * activation * (1.0f - activation));
}

// Launch inputs*outputs + outputs threads: one owner per weight or bias.
// online must be 0 or 1; online=1 requires count=1. Host validates dimensions.
extern "C" __global__ void update_fp32(
    const float* source, const float* delta, float* weights, float* biases,
    float* weight_velocity, float* bias_velocity, int inputs, int outputs,
    int count, float learning_rate, float momentum, int online) {
    const std::size_t index = thread_index();
    const std::size_t weight_count = static_cast<std::size_t>(inputs) * outputs;
    if (index >= weight_count + outputs) return;
    if (index < weight_count) {
        const int neuron = static_cast<int>(index / inputs);
        const int input = static_cast<int>(index % inputs);
        float scaled_gradient;
        if (online) {
            const float scale = learning_rate * delta[neuron];
            scaled_gradient = scale * source[input];
        } else {
            float gradient = 0.0f;
            for (int sample = 0; sample < count; ++sample) {
                gradient = fmaf(source[static_cast<std::size_t>(sample) * inputs + input],
                    delta[static_cast<std::size_t>(sample) * outputs + neuron], gradient);
            }
            scaled_gradient = (learning_rate / count) * gradient;
        }
        const float velocity = fmaf(momentum, weight_velocity[index], scaled_gradient);
        weight_velocity[index] = velocity;
        weights[index] += velocity;
    } else {
        const int neuron = static_cast<int>(index - weight_count);
        float scaled_gradient;
        if (online) {
            scaled_gradient = learning_rate * delta[neuron];
        } else {
            float gradient = 0.0f;
            for (int sample = 0; sample < count; ++sample) {
                gradient += delta[static_cast<std::size_t>(sample) * outputs + neuron];
            }
            scaled_gradient = (learning_rate / count) * gradient;
        }
        const float velocity = fmaf(momentum, bias_velocity[neuron], scaled_gradient);
        bias_velocity[neuron] = velocity;
        biases[neuron] += velocity;
    }
}

// Launch count*max(inputs, outputs) threads. Order is the JVM-generated shuffle.
extern "C" __global__ void gather_fp32(
    const float* all_inputs, const float* all_targets, const int* order,
    float* batch_inputs, float* batch_targets, int inputs, int outputs,
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


// Typed transport ABI 4. FP32 state/data remain float on the device and wire.
// Small fixed-shape entrypoints bound shared storage to the actual topology.
#define SMALL_FLOAT_ENTRY(Name, Parameters, Activations, Deltas) \
extern "C" __global__ void Name(float* packed, const float* inputs, const float* targets, \
    const int* orders, const int* topology, const int* active, int models, int layers, int samples, \
    int epochs, int batch_size, double learning_rate, double momentum, double beta, int mode, int online) { \
    small_train<float, float, Parameters, Activations, Deltas>(packed, inputs, targets, orders, topology, active, \
        models, layers, samples, epochs, batch_size, learning_rate, momentum, beta, mode, online); \
}
SMALL_FLOAT_ENTRY(small_train_fp32_packed, SMALL_PARAMETERS, SMALL_ACTIVATIONS, SMALL_DELTAS)
SMALL_FLOAT_ENTRY(small_train_fp32_2_4_1, 17, 7, 5)
SMALL_FLOAT_ENTRY(small_train_fp32_2_8_8_8_1, 177, 27, 25)
#undef SMALL_FLOAT_ENTRY

extern "C" __global__ void search_train_fp32_packed(float* packed, const float* inputs, const float* targets,
    const int* orders, const int* topology, const int* requests, float* checkpoint,
    int models, int samples, int batch_size, double learning_rate, double momentum, double beta, int mode, int online) {
    small_train<float, float>(packed, inputs, targets, orders, topology, nullptr, models, 0, samples,
        0, batch_size, learning_rate, momentum, beta, mode, online, requests, checkpoint);
}

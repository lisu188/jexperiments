__device__ double fastExpNegative(double value) {
    if (value <= -745.0) return 0.0;
    int exponent = (int)(value * 1.4426950408889634);
    double remainder = value - exponent * 0.6931471805599453;
    double square = remainder * remainder;
    double polynomial = 1.0 + remainder + square * (0.5 + remainder * (0.16666666666666666 +
        remainder * (0.041666666666666664 + remainder * 0.008333333333333333)));
    return ldexp(polynomial, exponent);
}
__device__ double activateValue(double value, int fastMode) {
    if (!fastMode) return 1.0 / (1.0 + exp(-value));
    if (value >= 0.0) return 1.0 / (1.0 + fastExpNegative(-value));
    double exponential = fastExpNegative(value);
    return exponential / (1.0 + exponential);
}
extern "C" __global__ void gatherRows(const double* source, int width, const int* order,
                                       int start, int count, double* destination) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = count * width;
    if (index >= total) return;
    int row = index / width;
    int column = index - row * width;
    destination[index] = source[order[start + row] * width + column];
}
extern "C" __global__ void addBiasAndSigmoid(double* values, const double* biases,
                                              int batch, int width, double beta, int fastMode) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * width;
    if (index >= total) return;
    int output = index % width;
    values[index] = activateValue((values[index] + biases[output]) * beta, fastMode);
}
extern "C" __global__ void outputDelta(const double* targets, const double* activations,
                                        double* deltas, int elements, double beta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    double activation = activations[index];
    deltas[index] = (targets[index] - activation) * beta * activation * (1.0 - activation);
}
extern "C" __global__ void applySigmoidDerivative(double* deltas, const double* activations,
                                                   int elements, double beta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    double activation = activations[index];
    deltas[index] *= beta * activation * (1.0 - activation);
}
extern "C" __global__ void reduceBiasGradient(const double* deltas, double* gradient,
                                               int batch, int width) {
    int output = blockIdx.x * blockDim.x + threadIdx.x;
    if (output >= width) return;
    double sum = 0.0;
    for (int sample = 0; sample < batch; ++sample) sum += deltas[sample * width + output];
    gradient[output] = sum;
}
extern "C" __global__ void momentumUpdate(double* values, double* velocity, const double* gradient,
                                           int elements, double momentum, double scale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    double nextVelocity = momentum * velocity[index] + scale * gradient[index];
    velocity[index] = nextVelocity;
    values[index] += nextVelocity;
}

__device__ float fastExpNegativeFloat(float value) {
    if (value <= -103.0f) return 0.0f;
    int exponent = (int)(value * 1.4426950408889634f);
    float remainder = value - exponent * 0.6931471805599453f;
    float square = remainder * remainder;
    float polynomial = 1.0f + remainder + square * (0.5f + remainder * (0.16666666666666666f +
        remainder * (0.041666666666666664f + remainder * 0.008333333333333333f)));
    return ldexpf(polynomial, exponent);
}
__device__ float activateValueFloat(float value, int fastMode) {
    if (!fastMode) return 1.0f / (1.0f + expf(-value));
    if (value >= 0.0f) return 1.0f / (1.0f + fastExpNegativeFloat(-value));
    float exponential = fastExpNegativeFloat(value);
    return exponential / (1.0f + exponential);
}
extern "C" __global__ void gatherRowsFloat(const float* source, int width, const int* order,
                                            int start, int count, float* destination) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = count * width;
    if (index >= total) return;
    int row = index / width;
    int column = index - row * width;
    destination[index] = source[order[start + row] * width + column];
}
extern "C" __global__ void addBiasAndSigmoidFloat(float* values, const float* biases,
                                                   int batch, int width, float beta, int fastMode) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * width;
    if (index >= total) return;
    int output = index % width;
    values[index] = activateValueFloat((values[index] + biases[output]) * beta, fastMode);
}
extern "C" __global__ void outputDeltaFloat(const float* targets, const float* activations,
                                             float* deltas, int elements, float beta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float activation = activations[index];
    deltas[index] = (targets[index] - activation) * beta * activation * (1.0f - activation);
}
extern "C" __global__ void applySigmoidDerivativeFloat(float* deltas, const float* activations,
                                                        int elements, float beta) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float activation = activations[index];
    deltas[index] *= beta * activation * (1.0f - activation);
}
extern "C" __global__ void reduceBiasGradientFloat(const float* deltas, float* gradient,
                                                    int batch, int width) {
    int output = blockIdx.x * blockDim.x + threadIdx.x;
    if (output >= width) return;
    float sum = 0.0f;
    for (int sample = 0; sample < batch; ++sample) sum += deltas[sample * width + output];
    gradient[output] = sum;
}
extern "C" __global__ void momentumUpdateFloat(float* values, float* velocity, const float* gradient,
                                                int elements, float momentum, float scale) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float nextVelocity = momentum * velocity[index] + scale * gradient[index];
    velocity[index] = nextVelocity;
    values[index] += nextVelocity;
}

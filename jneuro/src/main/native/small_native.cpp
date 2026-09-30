#include "small_training.h"
#include <limits>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

EXPORT int jsmall_abi() { return 1; }
EXPORT int jsmall_avx2() {
#if defined(_M_X64)
    int registers[4];
    __cpuidex(registers, 1, 0);
    constexpr int required = (1 << 12) | (1 << 27) | (1 << 28); // FMA, OSXSAVE, AVX.
    if ((registers[2] & required) != required || (GetEnabledXStateFeatures() & 6) != 6) return 0;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#elif defined(__x86_64__) && defined(__GNUC__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

// No C++ exception crosses the FFM boundary. Status: 1 invalid ABI argument,
// 2 unexpected native failure, 3 non-finite state. Failed state is never published.
EXPORT int jsmall_train(double* state, const double* inputs, const double* targets, const int* orders,
    int models, int samples, int epochs, int batch, int online,
    double learning_rate, double momentum, double beta, int sigmoid, int allow_simd) noexcept {
    if (!state || !inputs || !targets || !orders || models <= 0 || samples <= 0 || epochs < 1 || epochs > 64 ||
        batch <= 0 || (online != 0 && online != 1) || (online && batch != 1) || (sigmoid != 0 && sigmoid != 1) ||
        !std::isfinite(learning_rate) || !std::isfinite(momentum) || !std::isfinite(beta)) return 1;
    try {
        for (std::size_t index = 0; index < static_cast<std::size_t>(models) * samples * epochs; ++index)
            if (orders[index] < 0 || orders[index] >= samples) return 1;
        SmallArguments args{state, inputs, targets, orders, models, samples, epochs, batch, online,
            learning_rate, momentum, beta, sigmoid};
        int model = 0;
        if (allow_simd && jsmall_avx2()) for (; model + 4 <= models; model += 4) train_avx2(args, model);
        for (; model < models; ++model) train_group<Scalar>(args, model);
        for (std::size_t index = 0; index < static_cast<std::size_t>(models) * parameter_count * 2; ++index)
            if (!std::isfinite(state[index])) return 3;
        return 0;
    } catch (...) { return 2; }
}

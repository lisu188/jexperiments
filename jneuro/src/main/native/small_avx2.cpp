#include "small_training.h"
#include <immintrin.h>

struct Avx2 {
    using Value = __m256d;
    static constexpr int lanes = 4;
    static Value set(double a) { return _mm256_set1_pd(a); }
    static Value load(const double* a) { return _mm256_loadu_pd(a); }
    static void store(double* a, Value b) { _mm256_storeu_pd(a, b); }
    static Value add(Value a, Value b) { return _mm256_add_pd(a, b); }
    static Value sub(Value a, Value b) { return _mm256_sub_pd(a, b); }
    static Value mul(Value a, Value b) { return _mm256_mul_pd(a, b); }
    static Value fma(Value a, Value b, Value c) { return _mm256_fmadd_pd(a, b, c); }
    static Value activate(Value value, int mode) {
        // Preserve the reference sigmoid rather than silently enabling fast math.
        double lanes[4]; store(lanes, value);
        for (double& lane : lanes) lane = Scalar::activate(lane, mode);
        return load(lanes);
    }
};

void train_avx2(const SmallArguments& args, int first_model) { train_group<Avx2>(args, first_model); }

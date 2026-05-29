#include <stddef.h>

#include "fht.h"

#if defined(_WIN32)
#define FFHT_EXPORT __declspec(dllexport)
#else
#define FFHT_EXPORT
#endif

static int is_power_of_two(int n) {
    return n > 0 && ((n & (n - 1)) == 0);
}

static int integer_log2(int n) {
    int log_n = 0;
    while ((1 << log_n) < n) {
        ++log_n;
    }
    return log_n;
}

FFHT_EXPORT int ffht_axis0_float(float *data, int n, int d) {
    int col = 0;
    int log_n = 0;

    if (data == NULL || d < 1 || !is_power_of_two(n)) {
        return -1;
    }

    log_n = integer_log2(n);
    for (col = 0; col < d; ++col) {
        int status = fht_float(data + ((ptrdiff_t)col) * n, log_n);
        if (status != 0) {
            return status;
        }
    }

    return 0;
}

FFHT_EXPORT int ffht_axis0_double(double *data, int n, int d) {
    int col = 0;
    int log_n = 0;

    if (data == NULL || d < 1 || !is_power_of_two(n)) {
        return -1;
    }

    log_n = integer_log2(n);
    for (col = 0; col < d; ++col) {
        int status = fht_double(data + ((ptrdiff_t)col) * n, log_n);
        if (status != 0) {
            return status;
        }
    }

    return 0;
}

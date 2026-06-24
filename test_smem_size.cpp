#include <cstdio>

// Minimal repro to check SharedMemoryPlan size
// We'll use the actual header

#include "csrc/sm90/decode/sparse_fp8/config.h"

using namespace flash_mla;
using namespace flash_mla::sm90;
using namespace flash_mla::sm90::decode;
using namespace flash_mla::sm90::decode::sparse_fp8;

int main() {
    printf("sizeof(SharedMemoryPlan) = %zu bytes = %.1f KB\n",
        sizeof(SharedMemoryPlan), sizeof(SharedMemoryPlan) / 1024.0);
    return 0;
}

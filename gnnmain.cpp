// ================= main.cpp =================
#include "graphsage.h"
#include "runtime.h"
#include <chrono>

static uint64_t rdtsc() {
    _mm_lfence();
    uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}

int main(int argc, char** argv) {
  int numThreads = argc > 1 ? atoi(argv[1]) : 8;
  int allocateSize = argc > 2 ? atoi(argv[2]) : 1;
  int stealSize =  argc > 3 ? atoi(argv[3]) : 1;
  Runtime<ArgUnion, Worker<ArgUnion, FuncType>> rt(numThreads, allocateSize, stealSize);
  rt.init();
  auto t1 = std::chrono::steady_clock::now();
  rt.func();
  auto t2 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
  printf("%f ms\n", ms);
  return 0;
}

// Test file for attention.h
#include "runtime.h"
#include "attention.h"

int main(int argc, char** argv) {
    int numThreads = argc > 1 ? atoi(argv[1]) : 4;
    Runtime<AttentionArgs, Worker<AttentionArgs, AttentionFuncType>> rt(numThreads);
    rt.run();
    return 0;
}
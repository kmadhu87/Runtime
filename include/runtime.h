// ================= runtime.h =================
#ifndef RUNTIME_H
#define RUNTIME_H
#include "wsqueue.h"

template <typename A, typename B>
class Runtime {
public:
    llvm::SmallVector<B*> workers;
    
    Runtime<A>(int numThreads);
    void run();
    void init();
};

#endif

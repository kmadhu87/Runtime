// ================= runtime.h =================
#ifndef RUNTIME_H
#define RUNTIME_H
#include "dag.h"
#include <queue>
#include <thread>
#include <mutex>
#include <iostream>

class Runtime {
public:
    Runtime(int numThreads);
    void run(Task* start);

private:
    std::vector<std::thread> workers;
    std::vector<WSDeque*> queues;
    std::atomic<bool> stop{false};
    std::atomic<int> remainingTasks{0};

    void workerLoop(int tid);
};

#endif

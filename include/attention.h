#ifndef _ATTENTION_H_
#define _ATTENTION_H_

#include "runtime.h"
#include "wsqueue.h"
#include <llvm/ADT/SmallVector.h>
#include <iostream>
#include <cmath>

enum AttentionFuncType:uint8_t{
    SPAWN = 0,    
    SYNC = 1,    
    EXIT = 2
};

struct AttentionGlobalData {
    static constexpr int seqLength = 128;
    static constexpr int headDim = 32;
    static constexpr int TS = 8;  
    static constexpr int TD = 8; 

    static constexpr int numRowTiles = (seqLength + TS - 1) / TS;
    static constexpr int numColTiles = (headDim + TD - 1) / TD;

    static float Q[seqLength][headDim];
    static float K[seqLength][headDim];
    static float V[seqLength][headDim];
    static float Output[seqLength][headDim];

    // Initialize global matrices
    static void initialize() {
        for (int i = 0; i < seqLength; ++i) {
            for (int j = 0; j < headDim; ++j) {
                Q[i][j] = 1.0f + i * headDim + j;
                K[i][j] = 2.0f + i * headDim + j;
                V[i][j] = 3.0f + i * headDim + j;
                Output[i][j] = 0.0f;
            }
        }
    }
};

// Initialize static members
float AttentionGlobalData::Q[AttentionGlobalData::seqLength][AttentionGlobalData::headDim];
float AttentionGlobalData::K[AttentionGlobalData::seqLength][AttentionGlobalData::headDim];
float AttentionGlobalData::V[AttentionGlobalData::seqLength][AttentionGlobalData::headDim];
float AttentionGlobalData::Output[AttentionGlobalData::seqLength][AttentionGlobalData::headDim];

struct AttentionArgs {
    int tileIdx;     
    int attentionScore;  
    int slot;
    Worker<AttentionArgs, AttentionFuncType>::Task* address;
};

template<>
void __attribute__((hot)) __attribute__((preserve_none)) __attribute__((always_inline))
Worker<AttentionArgs, AttentionFuncType>::spawn(int tileIdx, Worker<AttentionArgs, AttentionFuncType>::Task* address, int slot, bool lastProducer) {
    count++;
    using GD = AttentionGlobalData;

    int tileRow = tileIdx / GD::numColTiles;
    int tileCol = tileIdx % GD::numColTiles;

    int rowStart = tileRow * GD::TS;
    int rowEnd = std::min(rowStart + GD::TS, GD::seqLength);
    int colStart = tileCol * GD::TD;
    int colEnd = std::min(colStart + GD::TD, GD::headDim);

    float score = 0.0f;
    for (int i = rowStart; i < rowEnd; ++i) {
        for (int j = colStart; j < colEnd; ++j) {
            score += GD::Q[i][j] * GD::K[i][j];
        }
    }

    for (int i = rowStart; i < rowEnd; ++i) {
        for (int j = colStart; j < colEnd; ++j) {
            GD::Output[i][j] += score * GD::V[i][j];
        }
    }

    if (address != nullptr) {
        writeDataToFrameImpl(address, slot, tileIdx, lastProducer);
    }
}

template<>
void __attribute__((hot)) __attribute__((preserve_none)) __attribute__((always_inline))
Worker<AttentionArgs, AttentionFuncType>::sync(int scoreInt, Worker<AttentionArgs, AttentionFuncType>::Task* address, int tileIdx, int slot) {
    count++;
    using GD = AttentionGlobalData;
    float score = *reinterpret_cast<float*>(&scoreInt);

    int tileRow = tileIdx / GD::numColTiles;
    int tileCol = tileIdx % GD::numColTiles;

    int rowStart = tileRow * GD::TS;
    int rowEnd = std::min(rowStart + GD::TS, GD::seqLength);
    int colStart = tileCol * GD::TD;
    int colEnd = std::min(colStart + GD::TD, GD::headDim);

    for (int i = rowStart; i < rowEnd; ++i) {
        for (int j = colStart; j < colEnd; ++j) {
            GD::Output[i][j] += score * GD::V[i][j];
        }
    }

    if (address == nullptr) {
        std::cout << "Attention computation complete!\n";
        std::cout << "Sample output[0][0] = " << GD::Output[0][0] << "\n";
        for (int i = 0; i < workers.size(); i++) {
            exited_per_worker[i].store(true, std::memory_order_relaxed);
        }
    } else {
        writeSyncDataToFrameImpl(address, slot, tileIdx);
    }
}

template<>
void __attribute__((cold)) __attribute__((preserve_none)) __attribute__((always_inline))
Worker<AttentionArgs, AttentionFuncType>::exitTask(int scoreInt, int tileIdx) {
    count++;
    using GD = AttentionGlobalData;

    std::cout << "=== Attention Output ===\n";
    for (int i = 0; i < GD::seqLength; ++i) {
        for (int j = 0; j < GD::headDim; ++j) {
            std::cout << GD::Output[i][j] << " ";
        }
        std::cout << "\n";
    }

    for (int i = 0; i < workers.size(); i++) {
        exited_per_worker[i].store(true, std::memory_order_relaxed);
    }
}

template class Runtime<AttentionArgs, Worker<AttentionArgs, AttentionFuncType>>;

template<>
Runtime<AttentionArgs, Worker<AttentionArgs, AttentionFuncType>>::Runtime(int numThreads){
    AttentionGlobalData::initialize();

    for (int i = 0; i < numThreads; i++) {
        Worker<AttentionArgs, AttentionFuncType>* worker = new Worker<AttentionArgs, AttentionFuncType>(i);
        workers.push_back(worker);
    }

    for (int i = 0; i < numThreads; i++) {
        workers[i]->setWorkers(workers);
    }
}

template<>
void Runtime<AttentionArgs, Worker<AttentionArgs, AttentionFuncType>>::init() {
    using GD = AttentionGlobalData;
    int totalTiles = GD::numRowTiles * GD::numColTiles;

    auto* worker0 = (Worker<AttentionArgs, AttentionFuncType>*)workers[0];

    Worker<AttentionArgs, AttentionFuncType>::Task* exitTask = worker0->pool.getFrame();
    exitTask->funcType = AttentionFuncType::EXIT;
    exitTask->remainingInputs.store(totalTiles, std::memory_order_relaxed);
    exitTask->address = nullptr;
    exitTask->slot = 0;
    exitTask->args[0] = 0;
    exitTask->args[1] = 0;

    for (int tileIdx = 0; tileIdx < totalTiles; ++tileIdx) {
        Worker<AttentionArgs, AttentionFuncType>::Task* task = worker0->pool.getFrame();
        task->funcType = AttentionFuncType::SPAWN;
        task->args[0] = tileIdx;
        task->address = exitTask;
        task->slot = 0;
        task->remainingInputs.store(0, std::memory_order_relaxed);

        if (worker0->readyQueue.isLocalQueueFull()) {
            worker0->waitQueueMutex.lock();
            worker0->readyQueue.steal_push_back(task);
            worker0->waitQueueMutex.unlock();
        } else {
            worker0->readyQueue.local_push_back(task);
        }
    }
}

template<>
void Runtime<AttentionArgs, Worker<AttentionArgs, AttentionFuncType>>::run() {
    init();
    for (auto w : workers)
        w->start();
    for (auto w : workers)
        w->join();

    int total = 0;
    for (auto w : workers)
        total += w->count;
    int totalsteals = 0;
    for (auto w : workers)
        totalsteals += w->numsteals;
    int totaltransitivesteals = 0;
    for (auto w : workers)
        totaltransitivesteals += w->numtransitivesteals;
    int totalSquashes = 0;
    for (auto w : workers)
        totalSquashes += w->lastProducerSquash;

    std::cout << "total:" << total << ", steals:" << totalsteals
              << ", totaltransitivesteals:" << totaltransitivesteals
              << ", total squashes:" << totalSquashes << "\n";
}

#endif //_ATTENTION_H_
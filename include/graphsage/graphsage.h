#ifndef GRAPHSAGE_H
#define GRAPHSAGE_H
#include "gswsqueue.h"
#include "runtime.h"
#include <vector>
#include "cnpy.h"
#include <cstring>
#include <cassert>
using namespace std;

enum FuncType:uint8_t{
	SPAWN = 0,
	MESSAGE = 1,
	SPAWN_AGGREGATE = 2,
	SYNC_AGGREGATE = 3,
	END = 4
};

typedef struct Args{
}Args;

union ArgUnion;

typedef struct SpawnArgs:Args{
	int* rowPtr;
	int64_t numRows;
}SpawnArgs;

typedef struct MsgArgs:Args{
	int slot;
	Worker<ArgUnion, FuncType>::Task* aggregatorAddress;
}MsgArgs;

typedef struct SpawnAggrArgs:Args{
	int numInputs;
	Worker<ArgUnion, FuncType>::Task* endTaskAddress;
}SpawnAggrArgs;

typedef struct SyncAggrArgs:Args{
	float** neigborInput;
	int numInputs;
	Worker<ArgUnion, FuncType>::Task* endTaskAddress;
}SyncAggrArgs;

typedef union ArgUnion{
	MsgArgs msgArgs;
	SpawnAggrArgs spawnAggrArgs;
	SyncAggrArgs syncAggrArgs;
	SpawnArgs spawnArgs;
} ArgUnion;

template<> 
void __attribute__((hot)) __attribute__((preserve_none)) Worker<ArgUnion, FuncType>::spawn(int *rowPtr, int64_t numRows){
	count++;
	auto endAddress = workers[0]->createExitTask(rowPtr, numRows);
	/*std::vector<size_t> order;
	std::vector<int> neighbors;
	std::vector<int> shuffledNeighbors;
	for(int i = 0; i<numRows; i++){
		int numInputs = rowPtr[i+1] - rowPtr[i];
		order.push_back(i);
		neighbors.push_back(numInputs);
	}
	
	std::sort(order.begin(), order.end(),
          [&](size_t a, size_t b) {
              return neighbors[a] > neighbors[b];
          });*/
	createSpawnAggregateTask(rowPtr, numRows, endAddress);
}

inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

void __attribute__((always_inline)) inline init(float* data, uint32_t seed) {
    for (int i = 0; i < featurelen; i+=4) {
       uint32_t r = hash32(seed + i);
        data[i] = (r >> 8) * (1.0f / 16777216.0f);
        uint32_t s = hash32(seed + i);
        data[i+1] = (s >> 8) * (1.0f / 16777216.0f);
        uint32_t t = hash32(seed + i);
        data[i+2] = (t >> 8) * (1.0f / 16777216.0f);
        uint32_t v = hash32(seed + i);
        data[i+3] = (v >> 8) * (1.0f / 16777216.0f);
    }
}

template<> 
void __attribute__((hot)) __attribute__((preserve_none)) Worker<ArgUnion, FuncType>::msgWithoutEnqueue(int slot, Worker<ArgUnion, FuncType>::Task* aggregatorAddress){
	count++;
	auto data = featureVectorPool.allocateFrame();
	::init(data, 1);
	writeMsgData(slot, aggregatorAddress, data);
} 

template<> 
void __attribute__((hot)) __attribute__((preserve_none)) Worker<ArgUnion, FuncType>::msg(int slot, Worker<ArgUnion, FuncType>::Task* aggregatorAddress, bool lastProducer){
	count++;
	auto data = featureVectorPool.allocateFrame();
	init(data, 1);
	writeMsgDataAndEnqueue(slot, aggregatorAddress, data, lastProducer);
} 

template<> 
void __attribute__((hot)) __attribute__((preserve_none)) Worker<ArgUnion, FuncType>::spawnAggr(int numInputs, Worker<ArgUnion, FuncType>::Task* exitTaskAddress){
	count++;
	createMessageAndSyncAggregateTasks(numInputs, exitTaskAddress);
}

volatile float sink;

template<> 
void __attribute__((hot)) __attribute__((preserve_none)) Worker<ArgUnion, FuncType>::syncAggr(float** neighborInput, int numInputs, Worker<ArgUnion, FuncType>::Task* endTaskAddress){
	count++;
	if(numInputs > 0){
		auto sum = featureVectorPool.allocateFrame();
		for(int i = 0; i<numInputs; i++){
			#pragma omp simd
			for(int j = 0; j<featurelen;j++){
				sum[j] += neighborInput[i][j];
			}
		}
		float inv = 1.0f / numInputs;
		#pragma omp simd
		for(int j = 0; j<featurelen;j++){
			sum[j] *= inv;
		}
		sink = sum[0];
		for(int i = 0; i<numInputs; i++){ 
			featureVectorPool.freeVec(neighborInput[i]);
		}
	}
	delete[] neighborInput;
	writeEndData(endTaskAddress);
} 

template<> 
void __attribute__((hot)) __attribute__((preserve_none)) inline Worker<ArgUnion, FuncType>::exitTask(){
	 count++;
	 for(int i = 0;i<workers.size(); i++){
	 	exited_per_worker[i].store(true, std::memory_order_relaxed);
	 }
} 

template class Runtime<ArgUnion, Worker<ArgUnion, FuncType>>;

template<>
Runtime<ArgUnion, Worker<ArgUnion, FuncType>>::Runtime(int numThreads, int allocateSize, int stealSize){
	for(int i = 0; i<numThreads; i++){
		Worker<ArgUnion, FuncType>* worker = new Worker<ArgUnion, FuncType>(i, allocateSize, stealSize);
		workers.push_back(worker);
	}
	
	for(int i = 0; i<numThreads; i++){
		workers[i]->setWorkers(workers);
	}
}

CSR* load_reddit_csr(const std::string& path) {
    auto npz = cnpy::npz_load(path);

    auto& indptr_arr  = npz.at("indptr");
    auto& indices_arr = npz.at("indices");
    auto& shape_arr   = npz.at("shape");

    // Types
    assert(indptr_arr.word_size == 4);   // int32
    assert(indices_arr.word_size == 4);  // int32

    int* rowptr = indptr_arr.data<int>();
    int* colind = indices_arr.data<int>();

    // shape is int64[2]
    const long long* shape = shape_arr.data<long long>();
    size_t nShape = shape[0];
    size_t ncols = shape[1];

    size_t mShape = indices_arr.shape[0];

    // Sanity check
    assert(indptr_arr.shape[0] == (size_t)(nShape + 1));
    assert(rowptr[nShape] == mShape);
    printf("indptr shape = %zu\n", indptr_arr.shape[0]);
    printf("indices shape = %zu\n", indices_arr.shape[0]);
    printf("shape = %lld %lld\n", shape[0], shape[1]);

    printf("rowptr[0] = %d\n", rowptr[0]);
    printf("rowptr[1] = %d\n", rowptr[1]);
    printf("rowptr[nShape-1] = %d\n", rowptr[nShape-1]);
    printf("rowptr[nShape] = %d\n", rowptr[nShape]);

    std::cout << "Loaded CSR: n=" << nShape << " m=" << mShape << "\n";

    int* rowPtr = new int[nShape + 1];
    int* colInd = new int[mShape];
    memcpy(rowPtr, rowptr, (nShape + 1) * sizeof(int));
    memcpy(colInd, colind, mShape * sizeof(int));
    return new CSR( rowPtr, colInd, nShape, mShape );
}


template<>
void Runtime<ArgUnion, Worker<ArgUnion, FuncType>>::init(){
    g = load_reddit_csr("/home/kavitha/compilerproject/data/reddit_adj.npz");
    assert(workers[0] != nullptr);
}

template<>
void Runtime<ArgUnion, Worker<ArgUnion, FuncType>>::func(){
    workers[0]->createSpawnTask(g->rowPtr, g->numNodes);
    for (auto w : workers) 
    	w->start();
    for (auto w : workers) 
    	w->join();
    int num = 0;
    for (auto w:workers){
    	num+=w->count;
    	std::cout<<w->workerId<<":"<<w->count<<"\n";
    }
    delete g;
    for(int i = 0; i<workers.size(); i++)
    	delete workers[i];
}


#endif

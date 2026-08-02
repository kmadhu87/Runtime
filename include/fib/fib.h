// ================= fib.h =================
#ifndef Fib_H
#define Fib_H
#include <atomic>
#include <condition_variable>
#include <functional>
#include <fibwsqueue.h>
#include <iostream>
#include "runtime.h"

enum FuncType:uint8_t{
	SPAWN = 0,
	SYNC = 1,
	EXIT
};

struct FibArgs{
	int left;
	int right;
	int slot;
	Worker<FibArgs, FuncType>::Task* address;
};

template<> 
void __attribute__((hot)) __attribute__((preserve_none)) __attribute__((always_inline)) Worker<FibArgs, FuncType>::spawn(int left, Worker<FibArgs, FuncType>::Task* address, int slot, bool lastProducer){
		//count++;
		if(left >= 2){
			//createFibChildrenAndLaunch(slot, address, left);
			auto syncTaskId = createNewSyncFrameCustom(slot, address, left - 2);
			createNewSpawnFrameAndWriteArgsAndLaunch(left - 1, syncTaskId, 0);
		}
		else {
  		    _mm_prefetch(&address->args, _MM_HINT_T0);		    		    		    		    
		    __builtin_prefetch(&address->remainingInputs, 1, 3);
		    writeDataToFrameImpl(address, slot, left, lastProducer);
		}
		return;
}

template<> 
void __attribute__((hot)) __attribute__((preserve_none)) __attribute__((always_inline))  Worker<FibArgs, FuncType>::sync(int left, Worker<FibArgs, FuncType>::Task* address, int right, int slot){
		//count++;
		int sum = left + right;		
		writeSyncDataToFrameImpl(address, slot, sum);
	   	return;
}

template<> 
void __attribute__((cold)) __attribute__((preserve_none)) __attribute__((always_inline)) Worker<FibArgs, FuncType>::exitTask(int left, int right){
		//count++;
		int sum = left + right;
   		std::cout<<"sum:"<<sum<<"\n";
   		for(int i = 0;i<workers.size(); i++){
	   		exited_per_worker[i].store(true, std::memory_order_relaxed);
	   	}
   		//std::atomic_thread_fence(std::memory_order_release);		
}

template class Runtime<FibArgs, Worker<FibArgs, FuncType> >;

template<>
Runtime<FibArgs, Worker<FibArgs, FuncType>>::Runtime(int numThreads){
	for(int i = 0; i<numThreads; i++){
		Worker<FibArgs, FuncType>* worker = new Worker<FibArgs, FuncType>(i);
		workers.push_back(worker);
	}
	
	for(int i = 0; i<numThreads; i++){
		workers[i]->setWorkers(workers);
	}
}

template<>
void Runtime<FibArgs, Worker<FibArgs,FuncType>>::init(){
    ((Worker<FibArgs, FuncType>*)workers[0])->createNewSpawnFrameAndWriteArgs(42, 0);
}

template<>
void Runtime<FibArgs, Worker<FibArgs, FuncType>>::run(){
    init();
    for (auto w : workers) 
    	w->start();
    for (auto w : workers) 
    	w->join();
    int total = 0;
    for(auto w:workers)
    	total+=w->count;
    int totalsteals = 0;
    for(auto w:workers)
    	totalsteals += w->numsteals;
   int totaltransitivesteals = 0;
    for(auto w:workers)
    	totaltransitivesteals += w->numtransitivesteals;
   int totalSquashes = 0;
    for(auto w:workers)
    	totalSquashes += w->lastProducerSquash;    	
    std::cout<<"total:"<<total<<", steals:"<<totalsteals<<", totaltransitivesteals:"<<totaltransitivesteals<<", total sqaushes:"<<totalSquashes<<"\n";
}


#endif

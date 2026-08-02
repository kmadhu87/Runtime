#ifndef Worker_H
#define Worker_H
#include <atomic>
#include <condition_variable>
#include <deque>
#include <emmintrin.h>
#include <functional>
#include <iostream>
#include <cassert>

#include <llvm/ADT/SmallVector.h>
#include <thread>
#include <cstdlib>
#define numframes 8192
#define n 65536*64 
#define m 512
#define featurelen 16 

alignas(64) std::atomic<int> exited_per_worker[8];

struct alignas(64) SpinLock {
  std::atomic<bool> locked{false};

  void lock() {
    for (;;) {
      // First: spin on shared reads
      while (locked.load(std::memory_order_relaxed)) {
        _mm_pause();
      }
      // Then: attempt to acquire
      if (!locked.exchange(true, std::memory_order_acquire)) {
        return;
      }
    }
  }

  void unlock() {
    locked.store(false, std::memory_order_release);
  }
};

template <typename Ty, typename FuncTy> struct alignas(64) Worker {
  int workerId;
  int count{0};
  int numsteals{0};
  int numtransitivesteals{0};
  int lastProducerSquash{0};
  
  llvm::SmallVector<Worker<Ty, FuncTy>*, 8> workers;

  int allocateSize;
  
  int stealSize;
  
  void setWorkers(llvm::SmallVector<Worker<Ty, FuncTy> *, 8> &workers) {
    this->workers = workers;
  }
  struct Task;

  typedef struct alignas(64) Task {
    alignas(64) std::atomic<int64_t> remainingInputs{0};
    bool lastProducer{false};
    bool expectLastProducer{false};   
    FuncTy funcType{FuncTy::MESSAGE};
    Ty args;    
    Task* next{nullptr};
  } Task;

template<typename B>
struct Block {
    B* ptr;
    Block* next;
};

  struct  alignas(64) TaskPool {
    Task* front[numframes];
    int frontIndex{numframes};
    Task* freePoolFront{nullptr};
    Block<Task>* allocatedPointers{nullptr};

    __attribute__((cold))
      Task* allocateFrame() {
        	Task *tasks = new Task[numframes]; 
		allocatedPointers = new Block{tasks, allocatedPointers};
		#pragma omp simd
        	for(int i = 0; i<numframes;i++)
        		front[i] = &tasks[i];
		Task* t = &tasks[0];
		t->next = nullptr;
		frontIndex = 1;
        	return t;
    }

    __attribute__((hot))    
      Task *getFrame() {
    	if(frontIndex < numframes){
	        return front[frontIndex++];
        }
        else if(freePoolFront){
		Task* t = freePoolFront;
		freePoolFront = freePoolFront->next;
		return t;
       	}
       	
	return allocateFrame();
    }
    
     void free(Task* t){
    	t->next = freePoolFront;
    	freePoolFront = t;
    	t->lastProducer = false;
        t->expectLastProducer = false;
    }
    
    ~TaskPool(){
	while(allocatedPointers != nullptr){
		auto temp = allocatedPointers;
		allocatedPointers = allocatedPointers->next;
		delete[] temp->ptr;
		delete temp;
	}
    }
  };
  TaskPool pool;
  
  struct  alignas(64) FeatureVectorPool {
     std::vector<float*> pool;
     
    __attribute__((hot))
     float* allocateFrame() {
     	if(pool.empty()){
		return new float[featurelen];
	}
	auto back = pool.back();
	pool.pop_back();
	return back;
     }

     void freeVec(float* t){
    	pool.push_back(t);
    }
    
    ~FeatureVectorPool(){
	for(auto t : pool)
		delete[] t;
    }
  };
  
  FeatureVectorPool featureVectorPool;
  
  struct  alignas(64) ReadyQueue {
	Task* readyLocalQueue[m];
  	int64_t localQueueBack{0};
	Task* readyStealQueue[n];
	int64_t front{0};
	int64_t back{0};
	
 	__attribute__((hot))
	bool   isLocalQueueFull(){
		return localQueueBack == m;
	}
  	__attribute__((hot))
  	  void local_push_back(Task* t){
		readyLocalQueue[localQueueBack++] = t;
  	}
  	
  	__attribute__((cold))
	  bool isLocalQueueEmpty(){
		return !localQueueBack;
	}
  	
  	__attribute__((hot))  	
  	  Task* local_pop_back(){
  		if(localQueueBack == 0)
  			return nullptr;
		auto t = readyLocalQueue[--localQueueBack];
		return t;
  	}
  	__attribute__((hot))
  	  void steal_push_back(Task* t){
		assert(back - front < n);
		readyStealQueue[back++&(n-1)] = t;
  	}
  	  	
  	__attribute__((hot))  	
  	  Task* steal_pop_front(){
  		if(back == front){
			return nullptr;
  		}	
		return readyStealQueue[front++&(n-1)];
  	}
  	
  	__attribute__((hot))  	
  	  std::vector<Task*>  steal_pop_front(int numtasks){
  		std::vector<Task*> stolen;
  		stolen.reserve(numtasks);
  		if(back == front)
  			return stolen;
		for(int i = 0; i< numtasks ;i++){
			if(front == back)
				break;
			stolen.push_back(readyStealQueue[front&(n-1)]);
			front++;
		}
		return stolen;
  	}
  	
  	__attribute__((hot))
  	  Task* steal_pop_back(){
  		if(back == front){
			return nullptr;
  		}
		return readyStealQueue[--back&(n-1)];
  	}
  };
  
    
  void __attribute__((hot)) __attribute__((preserve_none))   spawn(int *rowPtr, int64_t numRows);
  void __attribute__((hot)) __attribute__((preserve_none))   msgWithoutEnqueue(int slot, Task* aggregatorAddress);
  void __attribute__((hot)) __attribute__((preserve_none))   msg(int slot, Task* aggregatorAddress, bool lastProducer);  
  void __attribute__((hot)) __attribute__((preserve_none))   spawnAggr(int numInputs, Task* exitTaskAddress);
  void __attribute__((hot)) __attribute__((preserve_none))   syncAggr(float** neighborInput, int numInputs, Task* endTaskAddress);
  void __attribute__((hot)) __attribute__((preserve_none))   exitTask();
  
  std::thread thread;
  SpinLock waitQueueMutex;
  ReadyQueue readyQueue;

  Worker<Ty, FuncTy>(int workerId, int allocateSize, int stealSize) { 
  	this->workerId = workerId; 
  	this->allocateSize = allocateSize;
  	this->stealSize = stealSize;
  }
  
  void createSpawnTask(int *rowPtr, size_t numRows){
  	Task* spawnTask = pool.getFrame();
  	spawnTask->funcType  = FuncTy::SPAWN;
  	spawnTask->args.spawnArgs.rowPtr = rowPtr;
  	spawnTask->args.spawnArgs.numRows = numRows;
  	if(readyQueue.isLocalQueueFull()){ 	
		waitQueueMutex.lock();
		readyQueue.steal_push_back(spawnTask);
		waitQueueMutex.unlock();
	}else{
		readyQueue.local_push_back(spawnTask);
    	}
  }
  
    Task* createExitTask(int *rowPtr, size_t numRows){
  	Task* endTask = pool.getFrame();
	endTask->funcType = FuncTy::END;
	int numNonZeros = 0;
	for(size_t i = 0 ; i < numRows; i++){
		int numInputs = rowPtr[i+1] - rowPtr[i];
        	if(numInputs> 0)
        		numNonZeros++;
	}
	endTask->remainingInputs.store(numNonZeros, std::memory_order_relaxed);
	return endTask;
  }
  
  void inline createSpawnAggregateTask(int* rowPtr, size_t numNodes, Task* endAddress){
	size_t i = 0;
	if(numNodes > 1){
		for(; i < numNodes - 1; i++){
			if(readyQueue.isLocalQueueFull())
				break;
			if(rowPtr[i+1] - rowPtr[i]> 0){
  				auto spawnAggrTask = pool.getFrame();
				spawnAggrTask->funcType  = FuncTy::SPAWN_AGGREGATE;
				spawnAggrTask->args.spawnAggrArgs.numInputs = rowPtr[i+1] - rowPtr[i];
				spawnAggrTask->args.spawnAggrArgs.endTaskAddress = endAddress;
				readyQueue.local_push_back(spawnAggrTask);
			}
		}
		for(; i<numNodes - 1; i++){
			waitQueueMutex.lock();
			if(rowPtr[i+1] - rowPtr[i] > 0){	
				auto spawnAggrTask = pool.getFrame();
				spawnAggrTask->funcType  = FuncTy::SPAWN_AGGREGATE;
				spawnAggrTask->args.spawnAggrArgs.numInputs = rowPtr[i+1] - rowPtr[i];
				spawnAggrTask->args.spawnAggrArgs.endTaskAddress = endAddress;
				readyQueue.steal_push_back(spawnAggrTask);
			}
			waitQueueMutex.unlock();												
		}
	}
	if(i == numNodes - 1){
		if(rowPtr[i+1] - rowPtr[i] > 0){
			spawnAggr(rowPtr[i+1] - rowPtr[i], endAddress);
		}
	}
  }
  
  void inline createMessageAndSyncAggregateTasks(int numInputs, Task* endTask){
  	auto syncAggrTask = pool.getFrame();
	syncAggrTask->funcType  = FuncTy::SYNC_AGGREGATE;
	syncAggrTask->args.syncAggrArgs.neigborInput = new float*[numInputs];
	syncAggrTask->args.syncAggrArgs.numInputs = numInputs;
	syncAggrTask->args.syncAggrArgs.endTaskAddress = endTask;
	syncAggrTask->remainingInputs.store(numInputs, std::memory_order_relaxed);
	int localQueueSize = m - readyQueue.localQueueBack;	
	if(numInputs <= allocateSize + localQueueSize){
		syncAggrTask->expectLastProducer = true;
	}
	else{
		syncAggrTask->expectLastProducer = false;
	}
	int j = 0;
	int max = numInputs >= allocateSize + localQueueSize ? localQueueSize:(numInputs >= allocateSize ? numInputs - allocateSize : 0);
	bool enqueuedLocally = false;
        #pragma clang loop unroll_count(16)
	for( ;j < max; j++){
		Task* msgTask = pool.getFrame();
		assert(msgTask != nullptr);
		msgTask->args.msgArgs.slot =  j;
		msgTask->args.msgArgs.aggregatorAddress = syncAggrTask;
		msgTask->lastProducer = syncAggrTask->expectLastProducer  && j == 0;
		msgTask->funcType  = FuncTy::MESSAGE;				
		readyQueue.local_push_back(msgTask);
		enqueuedLocally = true;
	}
	bool enqueuedInStealQueue = false;
	max = numInputs - j > allocateSize? numInputs - allocateSize: 0;
        #pragma clang loop unroll_count(16)	
	for(int k = j ; k < max; k++){
		Task* msgTask = pool.getFrame();
		assert(msgTask != nullptr);
		msgTask->funcType  = FuncTy::MESSAGE;
		msgTask->args.msgArgs.slot =  j++;
		msgTask->args.msgArgs.aggregatorAddress = syncAggrTask;
		msgTask->lastProducer = false;
		waitQueueMutex.lock();		
		readyQueue.steal_push_back(msgTask);
		enqueuedInStealQueue = true;
		waitQueueMutex.unlock();		
	}
	assert(numInputs - j <= allocateSize);
	__builtin_prefetch(syncAggrTask->args.syncAggrArgs.neigborInput, 1);			
	if(numInputs - j > 1){
		if(syncAggrTask->expectLastProducer){
		        #pragma clang loop unroll_count(16)
			for(int k = j; k < numInputs - 1; k++){
				msgWithoutEnqueue(k, syncAggrTask);	
			}
		}else{
		        #pragma clang loop unroll_count(16)		
			for(int k = j; k < numInputs - 1; k++){
				msg(k, syncAggrTask, false);	
			}
		}
		j = numInputs - 1;
	}
	
	if(numInputs - j == 1)
		msg(j, syncAggrTask,  syncAggrTask->expectLastProducer && !enqueuedLocally);		
}

void inline writeMsgData(int slot, Task* aggregatorAddress, float* data){
	aggregatorAddress->args.syncAggrArgs.neigborInput[slot]  = data;
}

void inline writeMsgDataAndEnqueue(int slot, Task* aggregatorAddress, float* data, bool lastProducer){
	aggregatorAddress->args.syncAggrArgs.neigborInput[slot]  = data;
	if(!aggregatorAddress->expectLastProducer){
		if(aggregatorAddress->remainingInputs.fetch_sub(1, std::memory_order_relaxed) == 1){
			//if(readyQueue.isLocalQueueFull()){ 	    
				waitQueueMutex.lock();
				readyQueue.steal_push_back(aggregatorAddress);
				waitQueueMutex.unlock();
			//}else{
			//	readyQueue.local_push_back(aggregatorAddress);
    			//}
    		}
	}
	else if(lastProducer){
		if(readyQueue.isLocalQueueFull()){ 	    
			waitQueueMutex.lock();
			readyQueue.steal_push_back(aggregatorAddress);
			waitQueueMutex.unlock();
		}else{
			readyQueue.local_push_back(aggregatorAddress);
    		}
	}
}

void inline writeEndData(Task* exitTask){
	if(exitTask->remainingInputs.fetch_sub(1, std::memory_order_relaxed) == 1){
		if(readyQueue.isLocalQueueFull()){ 	    
			waitQueueMutex.lock();
			readyQueue.steal_push_back(exitTask);
			waitQueueMutex.unlock();
		}else{
			readyQueue.local_push_back(exitTask);
    		}
	}
}
	
  Task* executeLocalTask() {
   Task *t = readyQueue.local_pop_back();
   if(t) return t;
   waitQueueMutex.lock();
   t = readyQueue.steal_pop_back();
   waitQueueMutex.unlock();
   return t;
 }

 std::vector<Task*> stealRemoteTasks(int id, int numTasks){
    workers[id]->waitQueueMutex.lock();
    auto frames =  workers[id]->readyQueue.steal_pop_front(numTasks);
    workers[id]->waitQueueMutex.unlock();         
    return frames;
}

  __attribute__((hot, flatten)) void workerLoop() {
    while (true) {
      // try to pop from my readyQueue first
      Task* t = executeLocalTask();
      std::vector<Task*> tasks;
      if(t){
      	    FuncTy fn = t->funcType;
	    switch(fn){
	    	case FuncTy::SPAWN:{
			int * rowPtr = t->args.spawnArgs.rowPtr;
			int64_t numRows = t->args.spawnArgs.numRows;
	    	      	pool.free(t);
	    	      	spawn(rowPtr, numRows);
	    	      	break;
	    	}
	    	case FuncTy::MESSAGE:{
	        	Task* address = t->args.msgArgs.aggregatorAddress;
	        	int slot = t->args.msgArgs.slot;
	        	bool lastProducer = t->lastProducer;
			pool.free(t);
        	     	msg(slot, address, lastProducer);
        	     	break;
        	}
	       	case FuncTy::SPAWN_AGGREGATE:{
	        	int numInputs = t->args.spawnAggrArgs.numInputs;
	        	Task* endAddr = t->args.spawnAggrArgs.endTaskAddress;
			pool.free(t);        	      	
        	      	spawnAggr(numInputs, endAddr);
        	     	break;
        	}
       	       	case FuncTy::SYNC_AGGREGATE:{
	        	float** inputs = t->args.syncAggrArgs.neigborInput;
	        	int numInputs = t->args.syncAggrArgs.numInputs;
	        	Task* endTaskAddress = t->args.syncAggrArgs.endTaskAddress;
			pool.free(t);        	      	
		        syncAggr(inputs, numInputs, endTaskAddress);
        	     	break;
        	}
        	case FuncTy::END:{
			pool.free(t);        	      	        	      
        	      	exitTask();
        	     	return;
        	 }
       	     }
             continue;
      }
      else{
        int numthreads = workers.size();
        for (int i = 0; i  < numthreads; i++) {
       	   if(i != workerId){
        	  tasks = stealRemoteTasks(i, stealSize);
		  if(!tasks.empty()) {
		  	break;
		  }
	  }
        }
       }
        if(!tasks.empty()){
        	std::vector<Task*> taskList(tasks.begin(), tasks.begin()+(tasks.size() < allocateSize? tasks.size():allocateSize));
        	if(tasks.size() > allocateSize){
        		int i = allocateSize;
        		for(; i< tasks.size() && !readyQueue.isLocalQueueFull(); i++){
        			readyQueue.local_push_back(tasks[i]);
        		}
        		if(i<tasks.size()){
        			waitQueueMutex.lock();
	        		for(; i<tasks.size(); i++)
					readyQueue.steal_push_back(tasks[i]);
	           		waitQueueMutex.unlock();
	           	}
	   	}
	   	for(auto t: taskList){
      			FuncTy fn = t->funcType;
			switch(fn){
			    	case FuncTy::SPAWN:{
					int * rowPtr = t->args.spawnArgs.rowPtr;
					int64_t numRows = t->args.spawnArgs.numRows;
		    	      		pool.free(t);
		    	      		spawn(rowPtr, numRows);
		    	      		break;
			    	}
			    	case FuncTy::MESSAGE:{
			        	Task* address = t->args.msgArgs.aggregatorAddress;
			        	int slot = t->args.msgArgs.slot;
			        	bool lastProducer = t->lastProducer;
					pool.free(t);
		        	     	msg(slot, address, lastProducer);
		        	     	break;
		        	}
			       	case FuncTy::SPAWN_AGGREGATE:{
			        	int numInputs = t->args.spawnAggrArgs.numInputs;
			        	Task* endAddr = t->args.spawnAggrArgs.endTaskAddress;
					pool.free(t);        	      	
		        	      	spawnAggr(numInputs, endAddr);
		        	     	break;
		        	}
		       	       	case FuncTy::SYNC_AGGREGATE:{
			        	float** inputs = t->args.syncAggrArgs.neigborInput;
			        	int numInputs = t->args.syncAggrArgs.numInputs;
			        	Task* endTaskAddress = t->args.syncAggrArgs.endTaskAddress;
					pool.free(t);        	      	
		        	      	syncAggr(inputs, numInputs, endTaskAddress);
		        	     	break;
		        	}
		        	case FuncTy::END:{
					pool.free(t);        	      	        	      
		        	      	exitTask();
		        	     	return;
		        	 }
       	     		}
        	}
        	continue;
        }
        if(exited_per_worker[workerId].load(std::memory_order_relaxed)){
		break;
        }
    	std::this_thread::yield();
      }
    }
    
  void start() {
    thread = std::thread(&Worker::workerLoop, this);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(workerId, &cpuset);

    int r = pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t),
                                   &cpuset);
    if (r != 0) {
      perror("pthread_setaffinity_np");
    }
  }
  
 
  void   join() {
    if (thread.joinable())
      thread.join();
  }
};

#endif

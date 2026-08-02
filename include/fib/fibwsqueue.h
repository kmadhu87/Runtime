#ifndef Worker_H
#define Worker_H
#include <atomic>
#include <condition_variable>
#include <deque>
#include <emmintrin.h>
#include <functional>
#include <iostream>
#include <llvm/ADT/SmallVector.h>
#include <thread>
#include <vector>
#include <semaphore>
#include <queue>
#define numframes 64
#define m 4
#define n 65536*64

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
  struct Task;

  typedef struct alignas(64) Task {
    alignas(64) std::atomic<int32_t> remainingInputs{0};
    bool lastProducer{false};
    bool expectLastProducer{false};   
    int args[2];    
    FuncTy funcType;
    Task* __restrict__ address{nullptr};        
    int slot;
    Task* __restrict__ next;
    __attribute__((preserve_none))     __attribute__((hot))		
   void inline setValue(int index, int val){
	args[index] = val;
  }
	
  __attribute__((preserve_none))	
  void inline setAddress(int index, Task* val){
	address = val;
  }
} Task;

  struct  alignas(64) TaskPool {
    Task* front[numframes];
    int frontIndex{numframes};
    Task* freePoolFront[65536*64];
    int freePoolIndex{0};
    Task* allocatedPointers{nullptr};

    __attribute__((cold))
    inline Task* allocateFrame() {
        	Task *tasks = new Task[numframes];
        	tasks->next = allocatedPointers;
		allocatedPointers = tasks;
		#pragma omp simd
        	for(int i = 0; i<numframes;i++)
        		front[i] = &tasks[i];
		Task* t = &tasks[0];
		frontIndex = 1;
        	return t;
    }

    __attribute__((hot))    
    inline bool hasTwoFrames(){
    	return frontIndex < numframes - 1 || freePoolIndex >= 2;
    }

    __attribute__((hot))    
    inline std::pair<Task*, Task*> getTwoFrames(){
	if(frontIndex < numframes - 1){
        	auto  t = std::make_pair(front[frontIndex], front[frontIndex+1]);
        	frontIndex = frontIndex + 2;
        	return t;
        }else if(freePoolIndex >= 2){
        	auto ret = std::make_pair(freePoolFront[freePoolIndex-1], freePoolFront[freePoolIndex - 2]);
        	freePoolIndex -= 2;
        	return ret;
        }
        return std::make_pair(nullptr, nullptr);
    }
    
	
    __attribute__((hot))    
    inline Task *getFrame() {
    	if(frontIndex < numframes){
	        return front[frontIndex++];
        }
        else if(freePoolIndex > 0){
    		return freePoolFront[--freePoolIndex];
       	}
       	
	return allocateFrame();
    }
    
    inline void free(Task* t, bool enqueOnBack = false){
        if(freePoolIndex == 65536*64){
        	std::cout<<"free pool full, cant free\n";
        	exit(1);
        }
    	freePoolFront[freePoolIndex++] = t;
    }
    
    ~TaskPool(){
	while(allocatedPointers != nullptr){
		auto temp = allocatedPointers;
		allocatedPointers = allocatedPointers->next;
		delete[] temp;
	}
    }
  };
  TaskPool pool;
  
  struct  alignas(64) ReadyQueue {
	Task* readyLocalQueue[m];
  	int localQueueBack{0};

	Task* readyStealQueue[n];
	int front{0};
	int back{0};
	
 	__attribute__((hot))
	bool inline isLocalQueueFull(){
		return localQueueBack == m;
	}
  	__attribute__((hot))
  	void local_push_back(Task* t){
		readyLocalQueue[localQueueBack++] = t;
  	}
  	
  	__attribute__((cold))
	bool inline isLocalQueueEmpty(){
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
		if(back - front == n){
			std::cout<<"queue full\n";
			exit(0);
		}
		readyStealQueue[back++&(n-1)] = t;
  	}
  	
  	__attribute__((hot))  	
  	Task* steal_pop_front(){
  		if(back == front){
			return nullptr;
  		}	
		return readyStealQueue[front++&(n-1)];
  	}
  	
  	__attribute__((cold))  	
  	Task* steal_pop_back(){
  		if(back == front){
			return nullptr;
  		}
		return readyStealQueue[--back&(n-1)];
  	}
  };
  
  void __attribute__((preserve_none)) spawn(int, Task *, int, bool);
  void __attribute__((preserve_none)) sync(int, Task *, int, int);
  void __attribute__((preserve_none)) exitTask(int, int);

  ReadyQueue readyQueue;
  std::thread thread;
  SpinLock waitQueueMutex;

  llvm::SmallVector<Worker<Ty, FuncTy> *, 8> workers;

  Worker<Ty, FuncTy>(int workerId) { this->workerId = workerId; }

  void inline setWorkers(llvm::SmallVector<Worker<Ty, FuncTy> *, 8> &workers) {
    this->workers = workers;
  }

  void inline join() {
    if (thread.joinable())
      thread.join();
  }

 /* inline Task *createNewFrame(FuncTy fn, int numInputs) {
    __builtin_prefetch(pool.front, 1, 3);  
    Task* newTask = pool.getFrame();
    newTask->funcType = fn;
    newTask->addressOwner = workerId;
    newTask->remainingInputs.store(numInputs, std::memory_order_relaxed);
    newTask->expectLastProducer = false;
    newTask->lastProducer = false;    
    return newTask;
  }
  */
  
  __attribute__((always_inline)) __attribute__((hot))
  inline Task* createNewSyncFrameCustom(int val, Task* addr, int left) {
    Task* newTask = nullptr;
    Task* leftTask = nullptr;
    if(pool.hasTwoFrames()){
	auto tasks = pool.getTwoFrames();    
    	newTask = tasks.first;
    	leftTask = tasks.second;
    }else{
    	newTask = pool.getFrame();
    	leftTask = pool.getFrame();
    }
    __builtin_prefetch(&newTask->remainingInputs, 1, 3);
    newTask->funcType = addr?FuncTy::SYNC:FuncTy::EXIT;
    newTask->slot = val;
    newTask->address = addr;
    leftTask->funcType = FuncTy::SPAWN;
    leftTask->args[0] = left;
    leftTask->address = newTask;
    leftTask->slot = 1;
    if(readyQueue.isLocalQueueFull()){ 	   
	newTask->remainingInputs.store(2, std::memory_order_relaxed);
	newTask->expectLastProducer = false;
    	leftTask->lastProducer = false;
    	waitQueueMutex.lock();
	readyQueue.steal_push_back(leftTask);
        waitQueueMutex.unlock();
    }else{
    	if(left < 1){
    		newTask->expectLastProducer = true;
    		leftTask->lastProducer = true;
    		//lastProducerSquash++;
    	}else{
    		newTask->remainingInputs.store(2, std::memory_order_relaxed);
		newTask->expectLastProducer = false;
    		leftTask->lastProducer = false;
	}   
    	readyQueue.local_push_back(leftTask);
    } 
    return newTask;
  }

  __attribute__((always_inline)) __attribute__((hot))
  void inline createNewSpawnFrameAndWriteArgs(int left, int slot) {
    Task* newTask = pool.getFrame();
    newTask->funcType = FuncTy::SPAWN;
    newTask->args[0] = left;
    newTask->address = nullptr;
    newTask->slot = slot;
    if(readyQueue.isLocalQueueFull()){ 	    
    	waitQueueMutex.lock();
	readyQueue.steal_push_back(newTask);
        waitQueueMutex.unlock();
    }else{
    	readyQueue.local_push_back(newTask);
    }  	
  }
  
  __attribute__((always_inline)) __attribute__((hot))
  void inline createNewSpawnFrameAndWriteArgsAndLaunch(int left, Task* address, int slot){
  	spawn(left, address, slot, false);
  }

__attribute__((always_inline)) __attribute__((hot))
  void inline writeDataToFrameImpl(Task *task, int slot, int val, bool lastProducer) {
    task->setValue(slot, val);
    if(!task->expectLastProducer){
	if (task->remainingInputs.fetch_sub(1, std::memory_order_relaxed) == 1) {
  	_mm_prefetch(&readyQueue, _MM_HINT_T0);		    		    		    		    		
    	if(!readyQueue.isLocalQueueFull()){
		readyQueue.local_push_back(task);
		return;
    	}
    	else{
    		waitQueueMutex.lock();
		readyQueue.steal_push_back(task);
	        waitQueueMutex.unlock();
        	return;
        }
    }
  }else{
 	if(!lastProducer){
		return;
	}
    	else {
      	 _mm_prefetch(&readyQueue, _MM_HINT_T0);		    		    		    		    		
    	 if(!readyQueue.isLocalQueueFull()){
		readyQueue.local_push_back(task);
    		return;
    	}else{
    		waitQueueMutex.lock();
		readyQueue.steal_push_back(task);
        	waitQueueMutex.unlock();
        	return;
        }
    }
  }
 }

__attribute__((always_inline)) __attribute__((hot))
  void inline writeSyncDataToFrameImpl(Task *task, int slot, int val) {
    task->setValue(slot, val);
	if (task->remainingInputs.fetch_sub(1, std::memory_order_relaxed) == 1) {
  	_mm_prefetch(&readyQueue, _MM_HINT_T0);		    		    		    		    		
    	if(!readyQueue.isLocalQueueFull()){
		readyQueue.local_push_back(task);
		return;
    	}
    	else{
    		waitQueueMutex.lock();
		readyQueue.steal_push_back(task);
	        waitQueueMutex.unlock();
        	return;
        }
    }
 }
 
  void inline writeAddressToFrameImpl(Task *task, int slot, Task *val, bool enqueueLocally) {
    task->address = val;
    if (task->remainingInputs.fetch_sub(1, std::memory_order_relaxed) == 1) {
  	_mm_prefetch(&readyQueue, _MM_HINT_T0);		    		    		    		    		
    	if(enqueueLocally && !readyQueue.isLocalQueueFull()){
		readyQueue.local_push_back(task);
    		return;
    	}
    	waitQueueMutex.lock();
	readyQueue.steal_push_back(task);
        waitQueueMutex.unlock();
        return;
    }
  }
  
/*  void inline writeDataToFrame(Task *task, int slot, int val, bool local) {
    if (__builtin_expect(local, 1)) {
      writeDataToFrameImpl(task, slot, val);
    } else {
      workers[task->workerId]->writeDataToFrameImpl(task, slot, val);
    }
  }

  void inline writeAddressToFrame(Task *task, int slot, Task *val, bool local) {
    if (local) {
      writeAddressToFrameImpl(task, slot, val);
    } else {
      workers[task->workerId]->writeAddressToFrameImpl(task, slot, val);
    }
  }
  */
  
inline Task* executeLocalTask() {
   if(!readyQueue.isLocalQueueEmpty()){
   	auto task = readyQueue.local_pop_back();
   	return task;
   }
   waitQueueMutex.lock();   
   Task *t = readyQueue.steal_pop_back();
   waitQueueMutex.unlock();  
   return t;
}

inline std::vector<Task*> stealRemoteTasks(int id, int num){
    std::vector<Task*> tasks;
    workers[id]->waitQueueMutex.lock();
    for(int i = 0; i<num;i++){
    	Task* frameId =  workers[id]->readyQueue.steal_pop_front();
    	if(frameId == nullptr)
    		break;
    	tasks.push_back(frameId);
    	//if(frameId->addressOwner != id)
    	//	numtransitivesteals++;
    }
    //numsteals+=tasks.size();
     workers[id]->waitQueueMutex.unlock();         
    return tasks;
}

inline Task* stealRemoteTask(int id) {
    workers[id]->waitQueueMutex.lock();       
    Task* frameId =  workers[id]->readyQueue.steal_pop_front();
     workers[id]->waitQueueMutex.unlock();         
    return frameId;
  }

  __attribute__((hot, flatten)) void workerLoop() {
    while (true) {
      // try to pop from my readyQueue first
      Task* t = executeLocalTask();
      std::vector<Task*> tasks;
      if(t){
      	     FuncTy fn = t->funcType;
             int left = t->args[0];
       	     int right = t->args[1];
             Task* address = t->address;
             int slot = t->slot;
             bool lastProducer = t->lastProducer;
	     pool.free(t);
	     switch(fn){
	      case FuncTy::SPAWN:{
             	spawn(left, address, slot, lastProducer);
             	break;
             }
              case FuncTy::SYNC:{
              	sync(left, address, right, slot);            
             	break;
              }
              case FuncTy::EXIT:{
              	exitTask(left, right);
             	return;
              }
             } 
             continue;
      }
      else{
        int numthreads = workers.size();
        for (int i = 0; i  < numthreads; i++) {
       	   if(i != workerId){
          	_mm_prefetch(&workers[i]->waitQueueMutex, _MM_HINT_T0);
	          _mm_prefetch(&workers[i]->readyQueue, _MM_HINT_T0);     	
        	  tasks = stealRemoteTasks(i, 4*m);
		  if(!tasks.empty()) {
		  	break;
		  }
	  }
        }
       }
        if(!tasks.empty()){
           	Task * t = tasks[tasks.size() - 1];
           	if(tasks.size()>1){
			int count = 0;
			int i = tasks.size() - 2;
           		for( ;count < m && i >=0; i--, count++){
           			readyQueue.local_push_back(tasks[i]);
           		}
           		if(i >= 0){
        	   		waitQueueMutex.lock();
		           	for(int j = 0 ; j<=i; j++){
					readyQueue.steal_push_back(tasks[j]);
        	   		}
		           	waitQueueMutex.unlock();
	           	}
           	}
           	FuncTy fn = t->funcType;
        	int left = t->args[0];
       		int right = t->args[1];
        	Task* address = t->address;
        	int slot = t->slot;
		bool lastProducer = t->lastProducer;        	
		pool.free(t);
	    	switch(fn){
		      case FuncTy::SPAWN:{
        	     	spawn(left, address, slot, lastProducer);
        	     	break;
        	     	}
	       	      case FuncTy::SYNC:{
        	      	sync(left, address, right, slot);            
        	     	break;
        	      }
        	      case FuncTy::EXIT:{
        	      	exitTask(left, right);
        	     	return;
        	      }
        	 }
             	//else{
             	//	exitTask(left, right);
             	//}
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
};

#endif

// ================= worker.h =================
#ifndef Worker_H
#define Worker_H
#include <atomic>
#include <functional>
#include <vector>
#include <thread>
#include <condition_variable>
#include <deque>
#include <map>
#include <llvm/ADT/SmallVector.h>

template <typename Ty, typename FuncTy>
struct Worker {
    typedef void (Worker<Ty, FuncTy>::*FuncPtr)(Ty);
    __attribute__((regcall)) void invoke(FuncTy funcType, Ty args);
    int workerId;
    std::thread thread;
    std::mutex waitQueueMutex;
    std::condition_variable cv;
    
    typedef struct Task{
	FuncTy funcType;
  	Ty args;
  	int workerId;
    }Task;
    
    std::unordered_map<Task*, int32_t> remainingInputs;
    std::deque<Task*> readyQueue;
    
    llvm::SmallVector<Worker<Ty, FuncTy>*> workers;
    
    Worker<Ty, FuncTy>(int workerId){
    	this->workerId = workerId;
    }
    
    void inline setWorkers(llvm::SmallVector<Worker<Ty, FuncTy>*>& workers){
    	this->workers = workers;
    }

    void inline join() {
        if (thread.joinable())
            thread.join();
    }
  
  inline Task* createNewFrame(FuncTy fn, int numInputs){
	Task* newTask = new Task(fn, Ty{}, workerId);
	{
		std::lock_guard<std::mutex> lk(waitQueueMutex);
		remainingInputs[newTask] = numInputs;
	}
  	return newTask;
  }

  void inline createNewFrameAndWriteArgs(FuncTy fn, Ty args){
	Task* newTask = new Task(fn, args, workerId);
	{
		std::lock_guard<std::mutex> lk(waitQueueMutex);
		readyQueue.push_back(newTask);
	}
  }
  
  /*void deleteFrame(int32_t frameId){
 	while(!taskMutex.try_lock()){
	}
 	taskFunctions[frameId] = nullptr;
 	taskMutex.unlock();
	while(!readyQueueMutex.try_lock()){
   	}
   	int i = -1;
   	for(i = 0; i<readyQueue.size();i++){
		if(readyQueue[i] == frameId)
		break;
	}
	if(i < readyQueue.size() && i >=0)
		readyQueue.erase(readyQueue.begin() + i);
  }*/

  void inline writeDataToFrameImpl(Task* task, int slot, int val){
        task->args.setValue(slot, val);
        {	
                std::lock_guard<std::mutex> lk(waitQueueMutex);
	   	if(remainingInputs[task]== 1){
			readyQueue.push_back(task);
			remainingInputs.erase(task);
 		}else{
	 		remainingInputs[task]--;
	 	}
 	}
  }
  
  void inline writeAddressToFrameImpl(Task* task, int slot, Task* val){
        task->args.setAddress(slot, val);
        {	
                std::lock_guard<std::mutex> lk(waitQueueMutex);
	   	if(remainingInputs[task]== 1){
			readyQueue.push_back(task);
			remainingInputs.erase(task);
 		}else{
	 		remainingInputs[task]--;
	 	}
 	}
  }
  
 void inline writeDataToFrame(Task* task, int slot, int val, bool local){
  	if(local){
  	 	writeDataToFrameImpl(task,  slot, val);
  	}else{
		workers[task->workerId]->writeDataToFrameImpl(task, slot, val);
  	}
  }
  
  void inline writeAddressToFrame(Task* task, int slot, Task* val, bool local){
    	if(local){
    	        writeAddressToFrameImpl(task,  slot, val);
  	}else{
		workers[task->workerId]->writeAddressToFrameImpl(task,  slot, val);
  	}
  }
  
  inline Task* stealTask(bool local){
    Task* frameId = nullptr;
    {
   	std::lock_guard<std::mutex> lk(waitQueueMutex);
    	if (!readyQueue.empty()) {
    		if(local){
        		frameId = readyQueue.back();
	        	readyQueue.pop_back();
        	}else{
         		frameId = readyQueue.front();
	        	readyQueue.pop_front();       		
        	}
        }
    }
    return frameId;
  }
  
  void inline workerLoop(){
      while(true){
	Task* t;
        // try to pop from my readyQueue first
        t = stealTask(true);
        if(t == nullptr) {
            for (int i = 0; i < workers.size(); i++) {
            	if(i == workerId)
            		continue;
                t = workers[i]->stealTask(false);
                if (t != nullptr){
        	        break;
                }
            }
         }

        if (t == nullptr) {
            // no work anywhere
            //std::this_thread::yield();
            continue;
        }

        (invoke)(t->funcType, t->args);
         workers[t->workerId]->cv.notify_all();
         delete t;
     }
  }
  
  void start() {
        thread = std::thread(&Worker::workerLoop, this);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(workerId, &cpuset);

       int r = pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
       if (r != 0) {
         perror("pthread_setaffinity_np");
  	}
  }

};

#endif

// ================= runtime.cpp =================
#include "runtime.h"
#include <atomic>
#include <functional>
#include <optional>


void pinThreadToCore(int coreId) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(coreId, &cpuset);

  pthread_t current = pthread_self();
  int r = pthread_setaffinity_np(current, sizeof(cpu_set_t), &cpuset);
  if (r != 0) {
    perror("pthread_setaffinity_np");
  }
}

Runtime::Runtime(int num_workers)  {
  for (int i = 0; i < num_workers; ++i) {
  	queues.push_back(new WSDeque());
  }
}

void Runtime::run(Task* root){
    stop.store(false, std::memory_order_release);
    auto init_push = [&](Task* t) {
    if (t->remaining_deps.load() == 0) {
            queues[0]->push(t);
        }
    };

    remainingTasks.store(1);
    init_push(root);
    remainingTasks.store(1, std::memory_order_relaxed);
    // Launch workers
    for (int i = 0; i < queues.size(); ++i) {
        workers.emplace_back([&, i]{
            workerLoop(i);
        });
    }


    while (remainingTasks.load(std::memory_order_acquire) > 0) {
    }
    
    stop.store(true, std::memory_order_release);
    for (auto &w : workers) if (w.joinable()) w.join();
}

void Runtime::workerLoop(int tid) {
     auto& local = queues[tid];

    while (!stop.load(std::memory_order_acquire)) {
        Task* t = nullptr;
        // First try local
        t = local->pop();
        if (!t) {
            // Try to steal
            for (size_t i = 0; i < queues.size(); i++) {
                if (i == tid) continue;
                t = queues[i]->steal();
                if (t) {
                	break;
                }
            }
        }

        if (!t) {
            // No work anywhere
	    std::this_thread::yield();
            continue;
        }

        // Execute task inline
        t->fn(t);
        for (Task* child : t->children) {
        	if(satisfy_dep(child, local))
                    remainingTasks.fetch_add(1, std::memory_order_relaxed);
       		}
     	remainingTasks.fetch_sub(1, std::memory_order_relaxed);
        queues[tid]->cv.notify_all();
    }
}

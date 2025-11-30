// ================= dag.h =================
#ifndef DAG_H
#define DAG_H
#include <atomic>
#include <condition_variable>
#include <functional>
#include <vector>
#include <iostream>

struct Task {
  std::function<void(Task*)> fn; // inlined task
  std::vector<Task*> children;  // IDs of children tasks
  std::vector<Task*> value_sources;  // IDs of children tasks
  int value;
  std::atomic<int> remaining_deps{0};
};

struct DAG {
  std::vector<Task*> tasks;
};

struct WSDeque {
  std::atomic<size_t> top{0};
  std::atomic<size_t> bottom{0};
  std::vector<Task *> buf;
  std::mutex m;
  std::condition_variable cv;

  WSDeque(size_t size = 1024) : buf(size) {}

  bool push(Task *t) {
    size_t b = bottom.load(std::memory_order_relaxed);
    buf[b % buf.size()] = t;
    bottom.store(b + 1, std::memory_order_release);
    return true;
  }

  // safe pop: owner thread LIFO
  Task *pop() {
    size_t b = bottom.load(std::memory_order_relaxed);
    if (b == 0)
      return nullptr; // <-- avoid underflow
    b = b - 1;
    bottom.store(b, std::memory_order_seq_cst);

    size_t t = top.load(std::memory_order_seq_cst);
    if (t > b) {
      // empty
      bottom.store(b + 1, std::memory_order_relaxed);
      return nullptr;
    }

    Task *task = buf[b % buf.size()];
    if (t != b) {
      return task;
    }

    size_t expected = t;
    if (!top.compare_exchange_strong(expected, t + 1, std::memory_order_seq_cst,
                                     std::memory_order_relaxed)) {
      bottom.store(b + 1, std::memory_order_relaxed);
      return nullptr;
    }
    bottom.store(b + 1, std::memory_order_relaxed);
    return task;
  }

  Task *steal() {
    size_t t = top.load(std::memory_order_acquire);
    size_t b = bottom.load(std::memory_order_acquire);

    if (t >= b)
      return nullptr;
    Task *task = buf[t % buf.size()];
    if (!top.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst))
      return nullptr;
    return task;
  }
};

// Add edge: parent -> child
inline void add_edge(Task* parent, Task* child) {
    child->remaining_deps++;
    parent->children.push_back(child);
    child->value_sources.push_back(parent);
}

static std::pair<Task *, Task*> build_fib_dag(int n, std::vector<std::unique_ptr<Task>> &arena) {
  if (n < 2) {
    auto *leaf_sync =
        arena.emplace_back(std::make_unique<Task>()).get();
    leaf_sync->fn = [n](Task *t) {
      if(n == 1) t->value = 1;
      else if(n == 0) t->value = 0;
    };
    return std::make_pair(leaf_sync, leaf_sync);
  }

  auto *spawnN =
      arena.emplace_back(std::make_unique<Task>()).get();
  spawnN->value = n;

  auto *syncN =
      arena.emplace_back(std::make_unique<Task>()).get();

  auto left_sync = build_fib_dag(n - 1, arena);

  auto right_sync = build_fib_dag(n - 2, arena);

  add_edge(left_sync.second, syncN);
  add_edge(right_sync.second, syncN);

  add_edge(spawnN, left_sync.first);
  add_edge(spawnN, right_sync.first);

  spawnN->fn = [](Task *t) {  };

  syncN->fn = [](Task* t) {
        int sum = 0;
        for (Task* c : t->value_sources) {
            sum += c->value;
        }
        t->value = sum;
        std::cout<<"t value:"<<t->value<<"\n";
  };
  
  return std::make_pair(spawnN, syncN);
}

inline bool satisfy_dep(Task* t, WSDeque* q) {
    if (t->remaining_deps.fetch_sub(1) ==1) {
        q->push(t);
        return true;
    }
    return false;
}
#endif

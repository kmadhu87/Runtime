// ================= main.cpp =================
#include "runtime.h"
#include <iostream>

int main() {
  DAG dag;
  std::vector<Task *> owned;
  std::vector<std::unique_ptr<Task>> arena;
  auto root = build_fib_dag(12, arena);
  owned.push_back(std::move(root.first)); // owns the Task memory

  Runtime rt(16);
  rt.run(root.first);
  return 0;
}

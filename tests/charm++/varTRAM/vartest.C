#include <numeric>
#include <cassert>
#include <chrono>
#include <random>
#include <string>
#include <tuple>
#include <map>
#include <vector>
#include "pup_stl.h"
#include "vartest.decl.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int n_engines;
/* readonly */ int msg_count;
/* readonly */ int lambda;

class Main : public CBase_Main {
  CProxy_Engine engines;
  int generated_sum;
  int received_sum;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

public:
  Main(CkArgMsg* args) {
    if (args->argc == 4) {
      mainProxy = thisProxy;
      n_engines = std::atoi(args->argv[1]);
      msg_count = std::atoi(args->argv[2]);
      lambda = std::atoi(args->argv[3]);

      engines = CProxy_Engine::ckNew(n_engines);
      start_time = std::chrono::high_resolution_clock::now();
      engines.simulate();
    } else {
      CkPrintf("Usage: %s [n_engines] [msg_count] [lambda]\n", args->argv[0]);
      CkExit();
    }
    delete args;
  }

  void generatedSum(int val) {
    generated_sum = val;
  }

  void receivedSum(int received_sum) {
    CkPrintf("Generated sum: %d, received sum: %d (two sums should match)\n",
        generated_sum, received_sum);
    if (generated_sum != received_sum) {
      CkAbort("Sums did not match");
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> dur = end_time - start_time;
    CkPrintf("Time: %.3lf us\n", dur.count());
    CkExit();
  }
};

class Engine : public CBase_Engine {
  std::mt19937 mt;
  int iter;
  int generated;
  int received;

public:
  Engine() : mt(thisIndex), iter(0), generated(0), received(0) {}

  void simulate() {
    for (int i = 0; i < n_engines-1; i++) {
      for (int j = 0; j < msg_count; j++) {
        // Generate a random vector and send to a different engine
        auto temp = randVec();
        thisProxy[(thisIndex + 1 + i) % n_engines].ping(temp);

        // Store the sum of the generated vector elements
        generated = std::accumulate(temp.begin(), temp.end(), generated);
      }
    }
    contribute(sizeof(int), &generated, CkReduction::sum_int,
        CkCallback(CkReductionTarget(Main, generatedSum), mainProxy));
  }

  void ping(std::vector<int> vec) {
    received = std::accumulate(vec.begin(), vec.end(), received);
    if (++iter == (n_engines-1) * msg_count) {
      contribute(sizeof(int), &received, CkReduction::sum_int,
          CkCallback(CkReductionTarget(Main, receivedSum), mainProxy));
    }
  }

  std::vector<int> randVec() {
    std::uniform_int_distribution<> gen1(lambda/8, 3*lambda/8);
    std::uniform_int_distribution<> gen2(0, 373);

    // Generate vector with random size between lambda/8 and 3*lambda/8
    std::vector<int> gener(gen1(mt));
    for (auto& g : gener) {
      // Generate random number between 0 and 373
      g = gen2(mt);
    }
    return gener;
  }
};

#include "vartest.def.h"

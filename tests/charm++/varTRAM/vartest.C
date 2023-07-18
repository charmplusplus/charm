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

/* readonly */ CProxy_main mainProxy;

class main : public CBase_main {
  CProxy_engine engines;
  int N;
  int msgcount;
  int lambda;
  int startval;
  std::chrono::time_point<std::chrono::high_resolution_clock> starttime;

public:
  main(CkArgMsg* args) {
    if (args->argc == 4) {
      mainProxy = thisProxy;
      N = std::atoi(args->argv[1]);
      msgcount = std::atoi(args->argv[2]);
      lambda = std::atoi(args->argv[3]);
      engines = CProxy_engine::ckNew(N,msgcount,lambda,N);
      starttime = std::chrono::high_resolution_clock::now();
      engines.simulate();
    }
    else {
      CkPrintf("Usage: %s N Msgcount lambda [given: %d]\n",args->argv[0], args->argc);
      CkExit();
    }
    delete args;
  }

  void startsum(int val) {
    startval = val;
  }

  void done(int val) {
    auto endtime = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(endtime-starttime);
    assert(startval == val);
    long long timeval = (endtime - starttime).count();
    CkPrintf("Time : %lld us\n", timeval);
    CkExit();
  }
};

class engine : public CBase_engine {
  int N;
  int msgcount;
  int lambda;
  std::mt19937 mt;
  int iter;
  int result1,result2;

public:
  engine() {}

  engine(int _N, int _msgcount, int _lambda) : N(_N), msgcount(_msgcount), lambda(_lambda), mt(thisIndex), iter(0), result1(0), result2(0) {}

  void simulate() {
    for (int j = 0;j != N-1;++j) {
      for (int k = 0;k != msgcount;++k) {
        std::vector<int> temp;
        randvec(temp);
        thisProxy[(thisIndex+j+1)%N].ping(temp);
        result1 = std::accumulate(temp.begin(),temp.end(),result1);
      }
    }
    contribute(sizeof(int), &result1, CkReduction::sum_int, CkCallback(CkReductionTarget(main,startsum), mainProxy));
  }

  void ping(const std::vector<int>& val) {
    result2 = std::accumulate(val.begin(),val.end(),result2);
    ++iter;
    if (iter == (N-1)*msgcount) {
      contribute(sizeof(int), &result2, CkReduction::sum_int, CkCallback(CkReductionTarget(main,done), mainProxy));
    }
  }

  void randvec(std::vector<int>& gener) {
    //int length=lambda/4;
    std::uniform_int_distribution<> gen1(lambda/8,3*lambda/8);
    int length = gen1(mt);
    gener.resize(length);
    std::uniform_int_distribution<> gen2(0,373);
    for(auto& g : gener) {
      g = gen2(mt);
    }
  }
};

#include "vartest.def.h"

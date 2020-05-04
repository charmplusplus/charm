#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <tuple>
#include <map>
#include <vector>
struct msg;
template <typename T>
class is_PUPbytes;/*
template <>
struct is_PUPbytes<msg> {
  static const bool value = false;
};*/
#include "vartest.decl.h"
CProxy_main mainProxy; //readonly
struct msg {
  std::vector<int> data;
  msg(std::vector<int> s) : data(s) {}
  friend void operator|(PUP::er& er, msg& m) {
    er|m.data;
  }
  msg() : data() {}
};

class main : public CBase_main {
  CProxy_engine engines;
  int N;
  int msgcount;
  int lambda;
public:
  main(CkArgMsg* args) {
    if (args->argc == 4) {
      mainProxy=thisProxy;
      N=std::atoi(args->argv[1]);
      msgcount=std::atoi(args->argv[2]);
      lambda=std::atoi(args->argv[3]);
      engines=CProxy_engine::ckNew(N,msgcount,lambda,N);
      engines.simulate();
    }
    else {
      CkPrintf("Usage: %s N Msgcount lambda [given: %d]\n",args->argv[0], args->argc);
      CkExit();
    }
    delete args;
  }
  void startsum(int val) {
    CkPrintf("Sum of generated data: %d\n",val);
  }
  void done(int val) {
    CkPrintf("Sum of final data: %d\n",val);
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
    for (int j=0;j!=N-1;++j) {
      for (int k=0;k!=msgcount;++k) {
        auto temp = rand();
        thisProxy[(thisIndex+j+1)%N].ping(temp);
        result1 = std::accumulate(temp.data.begin(),temp.data.end(),result1);
      }
    }
    contribute(sizeof(int), &result1, CkReduction::sum_int, CkCallback(CkReductionTarget(main,startsum), mainProxy));
  } //after sending each message to the other PE
//send a self-message to add scheduling overhead(enforce msg count)
  void ping(msg val) {
    result2 = std::accumulate(val.data.begin(),val.data.end(),result2);
    ++iter;
    if (iter == (N-1)*msgcount) {
      contribute(sizeof(int), &result2, CkReduction::sum_int, CkCallback(CkReductionTarget(main,done), mainProxy));
    }
  }
  msg rand() {
    //int length=lambda/4;
    std::uniform_int_distribution<> gen1(lambda/8,3*lambda/8);
    int length = gen1(mt);
    std::uniform_int_distribution<> gen2(0,373);
    std::vector<int> gener;
    gener.reserve(gen1(mt));
    for (int j=0;j!=length;++j) {
      gener.push_back(gen2(mt));
    }
    return msg(gener);
    //return msg(0);
  }
};
#include "vartest.def.h"


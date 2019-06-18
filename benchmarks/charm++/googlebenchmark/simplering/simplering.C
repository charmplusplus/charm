#include <chrono>
#include <iostream>
#include "benchmark/benchmark.h"

#include "simplering.decl.h"


/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int RingSize;
/*readonly*/ int Rounds;

struct token_t : public CMessage_token_t
{
  int value;

  token_t ( const int value ) noexcept : value(value) { }
};


static void run(benchmark::State& state);

struct Main : public CBase_Main
{
  using dur_t = decltype( std::chrono::steady_clock::now() );

  Main(CkArgMsg* m)
  {
    RingSize = 100;
    Rounds = 1;

    if (m->argc > 1) RingSize = atoi(m->argv[1]);
    if (m->argc > 2) Rounds = atoi(m->argv[2]);

    benchmark::RegisterBenchmark("foo", run);
    benchmark::Initialize(&m->argc, m->argv);
    delete m;

    mainProxy = thisProxy;

    thisProxy.run_main();
  }

  void run_main()
  {
    benchmark::RunSpecifiedBenchmarks();
    CkExit();
  }
}; // Main


struct Node : public CBase_Node
{
  Node() { }
  Node(CkMigrateMessage*) { }
  ~Node() { }

  void take(token_t* t) const
  {
    if ( t->value-- )
    {
      thisProxy[(thisIndex + 1) % RingSize].take(t);
    }
    else
    {
      delete t;
    }
  }

}; // Node


static void run(benchmark::State& state)
{
  while ( state.KeepRunning() )
  {
    CProxy_Node ring = CProxy_Node::ckNew(RingSize);

    ring[0].take( new token_t{ Rounds * RingSize } );

    CkWaitQD ( );
    ring.ckDestroy();
    CkWaitQD ( );
  }
}


#include "simplering.def.h"


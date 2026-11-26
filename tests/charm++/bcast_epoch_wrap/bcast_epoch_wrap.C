// Array broadcasts across the wraparound of the broadcast epoch. Run with
// +bcastEpochBits 16 the epoch wraps every 65536 broadcasts, so the default
// 200000 broadcasts to each array cross it three times.
//
// Three arrays get the same stream, in bursts with a count check after each:
//   0: built with ckNew(n), anytime migration off (stable locations: a
//      broadcast is freed once every local element has it);
//   1: one element added with insert() (broadcasts are stored and freed by
//      the cleaning pass), the shape of charm issue #3860;
//   2: elements added with insert(), two per PE; each moves to the next PE
//      every 997 broadcasts (third argument; 0 for never), so elements catch up on stored broadcasts
//      across the wrap.
// Every element checks that the broadcasts arrive in order, none missing.
#include "bcast_epoch_wrap.decl.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int migratePeriod;  // broadcasts between moves of an array 2 element; 0: never

static constexpr int numArrays = 3;

class Main : public CBase_Main
{
  CProxy_Receiver arr[numArrays];
  int numElements[numArrays];
  int total = 200000;
  int burst = 1000;
  int sent = 0;
  int reports = 0;

public:
  Main(CkArgMsg* m)
  {
    if (m->argc > 1) total = atoi(m->argv[1]);
    if (m->argc > 2) burst = atoi(m->argv[2]);
    migratePeriod = m->argc > 3 ? atoi(m->argv[3]) : 997;
    delete m;
    mainProxy = thisProxy;

    numElements[0] = 2 * CkNumPes();
    CkArrayOptions opts(numElements[0]);
    opts.setAnytimeMigration(false).setStaticInsertion(true);
    arr[0] = CProxy_Receiver::ckNew(0, opts);

    numElements[1] = 1;
    arr[1] = CProxy_Receiver::ckNew();
    arr[1][0].insert(1);
    arr[1].doneInserting();

    numElements[2] = 2 * CkNumPes();
    arr[2] = CProxy_Receiver::ckNew();
    for (int i = 0; i < numElements[2]; i++) arr[2][i].insert(2, i % CkNumPes());
    arr[2].doneInserting();

    thisProxy.start();
  }

  void start() { sendBurst(); }

  void sendBurst()
  {
    for (int i = 0; i < burst && sent < total; i++, sent++)
      for (int a = 0; a < numArrays; a++) arr[a].hit(sent);
    reports = 0;
    for (int a = 0; a < numArrays; a++) arr[a].report();
  }

  void counted0(long count) { counted(0, count); }
  void counted1(long count) { counted(1, count); }
  void counted2(long count) { counted(2, count); }

  void counted(int which, long count)
  {
    if (count != (long)sent * numElements[which])
      CkAbort("bcast_epoch_wrap: array %d received %ld broadcasts in all, expected %ld",
              which, count, (long)sent * numElements[which]);
    if (++reports < numArrays) return;
    if (sent < total)
      sendBurst();
    else
    {
      CkPrintf("bcast_epoch_wrap: %d broadcasts to each of %d arrays delivered in order\n",
               total, numArrays);
      CkExit();
    }
  }
};

class Receiver : public CBase_Receiver
{
  int which = 0;
  int received = 0;

public:
  Receiver(int which_) : which(which_) {}
  Receiver(CkMigrateMessage*) {}

  void pup(PUP::er& p)
  {
    p | which;
    p | received;
  }

  void hit(int seq)
  {
    if (seq != received)
      CkAbort("bcast_epoch_wrap: array %d element %d got broadcast %d, expected %d",
              which, thisIndex, seq, received);
    received++;
    if (which == 2 && CkNumPes() > 1 && migratePeriod > 0 && received % migratePeriod == 0)
      migrateMe((CkMyPe() + 1) % CkNumPes());
  }

  void report()
  {
    long count = received;
    const int target = which == 0   ? CkReductionTarget(Main, counted0)
                       : which == 1 ? CkReductionTarget(Main, counted1)
                                    : CkReductionTarget(Main, counted2);
    contribute(sizeof(long), &count, CkReduction::sum_long, CkCallback(target, mainProxy));
  }
};

#include "bcast_epoch_wrap.def.h"

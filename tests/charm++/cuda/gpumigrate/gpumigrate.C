/**
 * Device-state migration test.
 *
 * Each element owns two device buffers holding patterns derived from its
 * index. The test moves every element to the next PE, several times, and after
 * each move reads the buffers back and compares them element by element. What
 * is under test is PUPMode::DEVICE and the migration path behind it: the
 * source packs the device buffers into a staged payload, that payload travels
 * on the device zerocopy path, and the destination copies it into the
 * element's freshly allocated buffers.
 *
 * Two buffers, not one, and the second is deliberately not a multiple of
 * DEVICE_PUP_ALIGN: a single buffer would not catch a packer and an unpacker
 * that disagree about where the next one starts.
 *
 * Migration is driven through AtSync with RotateLB, which moves every object one
 * PE onward at each step -- the strategy the manual recommends for exercising
 * pup routines and migration paths. Anytime migration (ckMigrate/migrateMe) is
 * deliberately NOT used: it is unsupported under CMK_GLOBAL_LOCATION_UPDATE,
 * which a GPU build needs (see "Global Location Update" in the manual). Driving
 * the move through a real load-balancing step is also closer to what this test
 * exists to protect.
 *
 * Verification runs in ResumeFromSync, which is reached once the element and its
 * device state have both landed, so the test needs no barrier of its own.
 *
 * Two PEs in one process covers the same-process transport. The IPC and RDMA
 * transports need more than one process and more than one physical node
 * respectively; see README.txt.
 */
#include "gpumigrate.decl.h"
#include "hapi.h"
#include <vector>

/* readonly */ CProxy_Main mainProxy;
/* readonly */ int numBlocks;
/* readonly */ int numRounds;

// Big enough that the payload is a real device allocation spanning several
// DEVICE_PUP_ALIGN boundaries, small enough to stay trivial.
#define BUF_LEN 4096
#define SECOND_LEN 517  // deliberately not a multiple of the device alignment

class Main : public CBase_Main
{
  CProxy_Block blocks;
  int round;
  int checked;

public:
  Main(CkArgMsg* m)
  {
    numBlocks = 4 * CkNumPes();
    numRounds = 3;
    if (m->argc > 1) numBlocks = atoi(m->argv[1]);
    if (m->argc > 2) numRounds = atoi(m->argv[2]);
    delete m;

    if (CkNumPes() < 2)
      CkAbort("gpumigrate needs at least 2 PEs: with one PE there is nowhere "
              "to migrate to and nothing would be tested.");

    round = 0;
    checked = 0;
    mainProxy = thisProxy;
    CkPrintf("gpumigrate: %d blocks over %d PEs, %d migration round(s), "
             "%d + %d device elements per block\n",
             numBlocks, CkNumPes(), numRounds, BUF_LEN, SECOND_LEN);

    blocks = CProxy_Block::ckNew(numBlocks);
    blocks.check();  // round 0: before anything has moved
  }

  void blockChecked()
  {
    if (++checked < numBlocks) return;
    checked = 0;

    if (round == numRounds)
    {
      CkPrintf("gpumigrate: PASSED -- device state survived %d migration "
               "round(s)\n", numRounds);
      CkExit();
      return;
    }
    round++;
    CkPrintf("gpumigrate: round %d, migrating every block one PE onward\n",
             round);
    blocks.step();
  }
};

class Block : public CBase_Block
{
  double* d_buf;
  int* d_second;
  int moves;   // how many times this element has migrated
  int lastPe;  // PE this element was on before the current step

public:
  Block() : d_buf(nullptr), d_second(nullptr), moves(0), lastPe(CkMyPe())
  {
    usesAtSync = true;
    allocate();
    fill();
  }

  Block(CkMigrateMessage* m)
      : CBase_Block(m), d_buf(nullptr), d_second(nullptr), moves(0), lastPe(-1)
  {
  }

  ~Block()
  {
    if (d_buf) hapiCheck(hapiFree(d_buf));
    if (d_second) hapiCheck(hapiFree(d_second));
  }

  void allocate()
  {
    hapiCheck(hapiMalloc((void**)&d_buf, BUF_LEN * sizeof(double)));
    hapiCheck(hapiMalloc((void**)&d_second, SECOND_LEN * sizeof(int)));
  }

  // The patterns are pure functions of the index, so they are checkable from
  // wherever the element lands.
  double expectedD(int i) const { return thisIndex * 1000.0 + i; }
  int expectedI(int i) const { return thisIndex * 7 + i; }

  void fill()
  {
    std::vector<double> h(BUF_LEN);
    for (int i = 0; i < BUF_LEN; i++) h[i] = expectedD(i);
    hapiCheck(cudaMemcpy(d_buf, h.data(), BUF_LEN * sizeof(double),
                         cudaMemcpyHostToDevice));

    std::vector<int> h2(SECOND_LEN);
    for (int i = 0; i < SECOND_LEN; i++) h2[i] = expectedI(i);
    hapiCheck(cudaMemcpy(d_second, h2.data(), SECOND_LEN * sizeof(int),
                         cudaMemcpyHostToDevice));
  }

  void pup(PUP::er& p)
  {
    CBase_Block::pup(p);
    p | moves;
    p | lastPe;

    // The device buffers must exist before they are pupped: PUPMode::DEVICE
    // copies into what the pointer names, it does not allocate.
    if (p.isUnpacking())
    {
      allocate();
      // Poison them first, or the test can pass without the device pup doing
      // anything: on the same-process path the source has just freed buffers
      // of exactly this size, and cudaMalloc is free to hand the very same
      // device memory back with the old contents still in it.
      hapiCheck(cudaMemset(d_buf, 0xA5, BUF_LEN * sizeof(double)));
      hapiCheck(cudaMemset(d_second, 0xA5, SECOND_LEN * sizeof(int)));
    }

    p(d_buf, BUF_LEN, PUP::PUPMode::DEVICE);
    p(d_second, SECOND_LEN, PUP::PUPMode::DEVICE);

    // The packing walker for a migration is deleting, and the copies it just
    // made are synchronous, so the source's buffers are finished with here.
    if (p.isDeleting())
    {
      hapiCheck(hapiFree(d_buf));
      d_buf = nullptr;
      hapiCheck(hapiFree(d_second));
      d_second = nullptr;
    }
  }

  void verify()
  {
    std::vector<double> h(BUF_LEN);
    hapiCheck(cudaMemcpy(h.data(), d_buf, BUF_LEN * sizeof(double),
                         cudaMemcpyDeviceToHost));
    for (int i = 0; i < BUF_LEN; i++)
    {
      if (h[i] != expectedD(i))
        CkAbort("gpumigrate: block %d on PE %d: d_buf[%d] is %f, expected %f "
                "(after %d migration(s))",
                thisIndex, CkMyPe(), i, h[i], expectedD(i), moves);
    }

    std::vector<int> h2(SECOND_LEN);
    hapiCheck(cudaMemcpy(h2.data(), d_second, SECOND_LEN * sizeof(int),
                         cudaMemcpyDeviceToHost));
    for (int i = 0; i < SECOND_LEN; i++)
    {
      if (h2[i] != expectedI(i))
        CkAbort("gpumigrate: block %d on PE %d: d_second[%d] is %d, expected "
                "%d (after %d migration(s))",
                thisIndex, CkMyPe(), i, h2[i], expectedI(i), moves);
    }
  }

  void check()
  {
    verify();
    mainProxy.blockChecked();
  }

  // Join the load-balancing step. RotateLB will move this element one PE on.
  void step() { AtSync(); }

  // Reached once this element and its device payload have both landed.
  void ResumeFromSync()
  {
    // A step in which nothing actually moved would let every later check pass
    // trivially -- the buffers were never packed, sent or unpacked, so of
    // course they still read correctly. Catch that here rather than report a
    // green run that tested nothing. RotateLB moves every object at every
    // step, so on more than one PE the destination is always a different one.
    if (CkMyPe() == lastPe)
      CkAbort("gpumigrate: block %d did not move at step %d (it is still on "
              "PE %d). Run with +balancer RotateLB -- without a balancer that "
              "migrates, this test verifies nothing.",
              thisIndex, moves + 1, CkMyPe());
    lastPe = CkMyPe();
    moves++;
    verify();
    mainProxy.blockChecked();
  }
};

#include "gpumigrate.def.h"

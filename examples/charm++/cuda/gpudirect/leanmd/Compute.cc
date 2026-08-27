#include "defs.h"
#include "Cell.h"
#include "Compute.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <unistd.h>

// CHARM_MD_ALLOCSTATS: how much of a migration step goes into device
// allocation. Migrated Computes drop their scratch in ~Compute and take it
// again on first use, and cudaMalloc/cudaFree both synchronize the device, so
// this is the suspected cost of moving an object at all.
namespace {
struct DevAllocStats {
  std::atomic<double> secs{0.0};
  std::atomic<long> calls{0};
  ~DevAllocStats() {
    if (calls.load() == 0) return;
    fprintf(stderr, "[md-allocstats] pid=%d device_alloc_free_time=%.3fs calls=%ld\n",
            (int)getpid(), secs.load(), calls.load());
    fflush(stderr);
  }
};
DevAllocStats g_dev_alloc;
inline bool allocStatsOn() {
  static const bool on = (getenv("CHARM_MD_ALLOCSTATS") != nullptr);
  return on;
}
// CHARM_MD_ASYNC_ALLOC: use CUDA's stream-ordered allocator. cudaMalloc and
// cudaFree both synchronize the whole device and serialize on a driver lock,
// which is what makes the first step after a migration so expensive -- a few
// hundred of them cost seconds. cudaMallocAsync/cudaFreeAsync draw from a
// pool and are ordered on the stream instead, so they neither synchronize nor
// contend. Safe here because every access to these buffers is already ordered
// against this chare's stream: the sender synchronizes it before a MEMCPY-mode
// pull, the IPC path waits on the staged event, and a balancing step now waits
// for the force-send acknowledgements before anything is released.
inline bool asyncAllocOn() {
  static const bool on = (getenv("CHARM_MD_ASYNC_ALLOC") != nullptr);
  return on;
}
inline hapiError_t mdDevMalloc(void** p, size_t n, cudaStream_t s) {
  if (asyncAllocOn() && s != NULL) return cudaMallocAsync(p, n, s);
  return hapiMalloc(p, n);
}
inline hapiError_t mdDevFree(void* p, cudaStream_t s) {
  if (asyncAllocOn() && s != NULL) return cudaFreeAsync(p, s);
  return hapiFree(p);
}

struct AllocTimer {
  double t0; bool on;
  AllocTimer() : t0(0), on(allocStatsOn()) { if (on) t0 = CkWallTimer(); }
  ~AllocTimer() {
    if (!on) return;
    double d = CkWallTimer() - t0;
    double cur = g_dev_alloc.secs.load();
    while (!g_dev_alloc.secs.compare_exchange_weak(cur, cur + d)) {}
    g_dev_alloc.calls.fetch_add(1, std::memory_order_relaxed);
  }
};
}  // namespace

extern /* readonly */ CProxy_StreamPool streamPool;

//compute - Default constructor
Compute::Compute() : stepCount(1), d_energyPartial(NULL), d_energyScalar(NULL),
                     h_energy(NULL) {
  energy[0] = energy[1] = 0;
  usesAtSync = true;
  cap[0] = cap[1] = 0;
  d_pos[0] = d_pos[1] = NULL;
  d_force[0] = d_force[1] = NULL;
  nPart[0] = nPart[1] = 0;
  stream = NULL;
  pendingForceSends = 0;
  deriveCells();
}

Compute::Compute(CkMigrateMessage *msg): CBase_Compute(msg) {
  usesAtSync = true;
  cap[0] = cap[1] = 0;
  d_pos[0] = d_pos[1] = NULL;
  d_force[0] = d_force[1] = NULL;
  d_energyPartial = NULL;
  d_energyScalar = NULL;
  h_energy = NULL;
  stream = NULL;
  pendingForceSends = 0;
  deriveCells();
  delete msg;
}

// NOTHING device-side may be touched from the constructors.
//
// Computes are inserted by Cell::createComputes, which runs from the main chare
// constructor -- during startup, before HAPI has necessarily selected this PE's
// device and before the stream pool group is guaranteed to have a branch here.
// A cudaMalloc from there fails, hapiCheck aborts, and because it happens ahead
// of any entry method the process dies without printing a single line.
//
// So every device resource is taken on first use instead, from the post entry
// method, which is the earliest point at which this chare does real work.
void Compute::ensureDevice() {
  if (stream != NULL) return;
  stream = streamPool.ckLocalBranch()->acquire();
  hapiCheck(mdDevMalloc((void**)&d_energyScalar, sizeof(double), stream));
  hapiCheck(hapiMallocHost((void**)&h_energy, sizeof(double)));
}

Compute::~Compute() { freeDevice(); }

// Recover the two cells from my own index. createComputes built it as
// (cell1 + KAWAY, cell2 + KAWAY) with cell2 possibly outside the array, so both
// halves need wrapping. Knowing this here is what lets the post entry method
// route an arriving message to the right slot without a section cookie.
void Compute::deriveCells() {
  cellA[0] = WRAP_X(thisIndex.x1 - KAWAY_X);
  cellA[1] = WRAP_Y(thisIndex.y1 - KAWAY_Y);
  cellA[2] = WRAP_Z(thisIndex.z1 - KAWAY_Z);
  cellB[0] = WRAP_X(thisIndex.x2 - KAWAY_X);
  cellB[1] = WRAP_Y(thisIndex.y2 - KAWAY_Y);
  cellB[2] = WRAP_Z(thisIndex.z2 - KAWAY_Z);
  selfCompute = (thisIndex.x1 == thisIndex.x2 && thisIndex.y1 == thisIndex.y2 &&
                 thisIndex.z1 == thisIndex.z2);
}

int Compute::slotFor(int cx, int cy, int cz) const {
  return (cx == cellA[0] && cy == cellA[1] && cz == cellA[2]) ? 0 : 1;
}

// Grow one slot's buffers. Safe to do from the post entry method because the slot
// is about to be overwritten by the message that triggered it: the only data at
// risk belongs to the same slot and is already spent.
void Compute::ensureSlot(int s, int n) {
  if (n <= cap[s]) return;
  AllocTimer _t;
  const int newcap = n + n / 4 + 64;

  if (d_pos[s]) hapiCheck(mdDevFree(d_pos[s], stream));
  if (d_force[s]) hapiCheck(mdDevFree(d_force[s], stream));
  hapiCheck(mdDevMalloc((void**)&d_pos[s], sizeof(vec3) * newcap, stream));
  hapiCheck(mdDevMalloc((void**)&d_force[s], sizeof(vec3) * newcap, stream));

  // The energy partials are indexed by the A-side atom, one entry per block.
  if (s == 0) {
    if (d_energyPartial) hapiCheck(mdDevFree(d_energyPartial, stream));
    hapiCheck(mdDevMalloc((void**)&d_energyPartial, sizeof(double) * newcap, stream));
  }
  cap[s] = newcap;
}

void Compute::freeDevice() {
  AllocTimer _t;
  for (int s = 0; s < 2; s++) {
    if (d_pos[s])   { hapiCheck(mdDevFree(d_pos[s], stream));   d_pos[s] = NULL; }
    if (d_force[s]) { hapiCheck(mdDevFree(d_force[s], stream)); d_force[s] = NULL; }
    cap[s] = 0;
  }
  if (d_energyPartial) { hapiCheck(mdDevFree(d_energyPartial, stream)); d_energyPartial = NULL; }
  if (d_energyScalar)  { hapiCheck(mdDevFree(d_energyScalar, stream));  d_energyScalar = NULL; }
  if (h_energy)        { hapiCheck(hapiFreeHost(h_energy));    h_energy = NULL; }
}

void Compute::calculateForces(int ref, int ord, int cx, int cy, int cz, int& n,
                              vec3*& pos, CkDeviceBufferPost* devicePost) {
  ensureDevice();
  const int s = slotFor(cx, cy, cz);
  ensureSlot(s, n);
  pos = d_pos[s];
  nPart[s] = n;
  ordinal[s] = ord;
  cellIdx[s][0] = cx; cellIdx[s][1] = cy; cellIdx[s][2] = cz;
  devicePost[0].hapi_stream = stream;
}

// The displacement to apply to B's positions so the two cells are adjacent across
// a periodic boundary.
//
// calcPairForces detected the wrap with abs(first - second) > 1 and shifted A by
// +diff. Shifting B by -diff gives the same separation and leaves the position
// arrays alone, which is required here: they are device resident and read by
// other computes at the same time.
vec3 Compute::periodicShift() const {
  vec3 shift(0.0);
  if (abs(cellA[0] - cellB[0]) > 1)
    shift.x = (cellB[0] < cellA[0]) ?  (double)CELL_SIZE_X * cellArrayDimX
                                    : -(double)CELL_SIZE_X * cellArrayDimX;
  if (abs(cellA[1] - cellB[1]) > 1)
    shift.y = (cellB[1] < cellA[1]) ?  (double)CELL_SIZE_Y * cellArrayDimY
                                    : -(double)CELL_SIZE_Y * cellArrayDimY;
  if (abs(cellA[2] - cellB[2]) > 1)
    shift.z = (cellB[2] < cellA[2]) ?  (double)CELL_SIZE_Z * cellArrayDimZ
                                    : -(double)CELL_SIZE_Z * cellArrayDimZ;
  return shift;
}

// Measurement window, in steps before an AtSync, mirroring pic2d.
#define LB_INSTRUMENT_WINDOW 3

// Instrumentation is only useful for the steps the balancer will actually read,
// and it is not free: every traced kernel becomes a CUPTI activity record that
// hapiProcessCuptiBuffers has to walk at LB time. Tracing all 90 steps of a run
// to balance on the last few costs seconds per LB event -- far more than the
// imbalance being corrected. Toggle per PE (not per chare: hundreds of Computes
// share a PE and the switch is PE-wide) and only on a transition.
void Compute::updateInstrumentation() {
  static thread_local bool instrumenting = true;  // runtime default at startup
  static thread_local int lastStep = -1;
  if (lastStep == stepCount) return;              // first Compute of the step wins
  lastStep = stepCount;

  int nextLb;
  if (stepCount <= firstLdbStep) {
    nextLb = firstLdbStep;
  } else {
    const int k = (stepCount - firstLdbStep + ldbPeriod - 1) / ldbPeriod;
    nextLb = firstLdbStep + k * ldbPeriod;
  }

  const bool want = (nextLb - stepCount) <= LB_INSTRUMENT_WINDOW;
  if (want != instrumenting) {
    instrumenting = want;
    if (want) LBTurnInstrumentOn(); else LBTurnInstrumentOff();
  }
}

void Compute::launchForces() {
  updateInstrumentation();
  const double cutoffSq = (double)PTP_CUT_OFF * (double)PTP_CUT_OFF;
  const bool doEnergy = (stepCount == 1 || stepCount == finalStepCount);
  const int nA = nPart[0];

  invokeZeroForces(d_force[0], nA, stream);

  if (selfCompute) {
    invokePairForce(d_pos[0], nA, d_pos[0], nA, d_force[0], vec3(0.0), cutoffSq,
                    true, doEnergy ? d_energyPartial : NULL, stream);
  } else {
    const int nB = nPart[1];
    invokeZeroForces(d_force[1], nB, stream);

    const vec3 shiftB = periodicShift();
    // Forces on A from B, carrying the energy for the whole pair...
    invokePairForce(d_pos[0], nA, d_pos[1], nB, d_force[0], shiftB, cutoffSq,
                    false, doEnergy ? d_energyPartial : NULL, stream);
    // ...and the reciprocal, with the roles and the shift both reversed. Newton's
    // third law would halve this work, but only by writing into the other cell's
    // force array under atomics; recomputing is cheaper than the contention.
    const vec3 shiftA(-shiftB.x, -shiftB.y, -shiftB.z);
    invokePairForce(d_pos[1], nB, d_pos[0], nA, d_force[1], shiftA, cutoffSq,
                    false, NULL, stream);
  }

  if (doEnergy) {
    invokeReduceDoubles(d_energyPartial, nA, d_energyScalar, stream);
    hapiCheck(cudaMemcpyAsync(h_energy, d_energyScalar, sizeof(double),
                              cudaMemcpyDeviceToHost, stream));
  }

  CkCallback* cb = new CkCallback(CkIndex_Compute::forcesReady(),
                                  thisProxy[thisIndex]);
  hapiAddCallback(stream, cb);
}

// Ship each force array straight back to the cell it belongs to, tagged with the
// ordinal that cell handed out, so it lands in its own slot and is folded in by a
// kernel. This is the replacement for contributing into the cell's section
// reduction.
// A device zerocopy send is a pull: the cell reads d_force out of this chare's
// memory some time after the send returns. Nothing else keeps that memory
// alive or unchanged -- the next step's invokeZeroForces rewrites it, and on a
// balancing step ~Compute frees it outright -- so every send carries a
// completion callback and run() drains them at the end of each step before
// either can happen.
//
// This used to attach the callback only on balancing steps, relying on the fact
// that a staged cross-process send copies d_force into the runtime's
// communication buffer on this chare's own stream, which orders the next step's
// kernels behind the copy for free. That is a property of the staged transport,
// not of the API: the direct CUDA IPC transport hands the cell this buffer
// itself and leaves the completion callback as the only ordering there is.
void Compute::sendForces() {
  if (stepCount == 1) energy[0] = *h_energy;
  else if (stepCount == finalStepCount) energy[1] = *h_energy;

  pendingForceSends = selfCompute ? 1 : 2;

  CkCallback sentCb(CkIndex_Compute::forceSendDone(), thisProxy[thisIndex]);

  cellArray(cellIdx[0][0], cellIdx[0][1], cellIdx[0][2])
      .receiveForces(stepCount, ordinal[0], nPart[0],
                     CkDeviceBuffer(d_force[0], sentCb, stream));

  if (!selfCompute)
    cellArray(cellIdx[1][0], cellIdx[1][1], cellIdx[1][2])
        .receiveForces(stepCount, ordinal[1], nPart[1],
                       CkDeviceBuffer(d_force[1], sentCb, stream));
}

//pack important information if I am moving
void Compute::pup(PUP::er &p) {
  CBase_Compute::pup(p);
  __sdag_pup(p);
  p | stepCount;
  p | pendingForceSends;
  PUParray(p, energy, 2);

  // Nothing device-side is puped. The scratch holds nothing that outlives a step
  // -- AtSync is called between steps -- so the destructor releases it on the old
  // PE and the migration constructor starts empty, reallocating on first use.
  // Freeing here instead would misfire under PUP::sizer, which also reports
  // isPacking().
}

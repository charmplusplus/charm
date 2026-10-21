/* Migratable user-level threads: a round trip through serialization.
 *
 * A migratable thread keeps its stack and its heap inside an Isomalloc
 * context, so that both land at the same virtual addresses wherever the thread
 * is resumed and every pointer it holds -- into its own stack, into its own
 * heap -- stays valid. This test exercises that end to end without a network:
 * the thread is packed out of the process and unpacked back into it, which is
 * the migration path minus the send.
 *
 * What would fail if the machinery were wrong:
 *   - a stack allocated by malloc: the unpacked context maps somewhere else
 *     and the resumed thread returns into freed memory;
 *   - a heap outside the context: the thread's own pointers dangle after the
 *     round trip;
 *   - a mis-pupped machine context: the resume jumps to nothing.
 */

#include "migthread.decl.h"

#include "memory-isomalloc.h"
#include "pup.h"

#include <cstring>
#include <vector>

CProxy_Main mainProxy;

namespace {

constexpr int kPayload = 4096;
constexpr int kNumThreads = 4;

struct ThreadState {
  int index;
  /* Allocated from the thread's own Isomalloc heap while its context is
     active, so it travels with the thread. */
  unsigned char *payload;
  /* A pointer into the thread's own stack, recorded before the round trip and
     checked after it. */
  int *stackWitness;
  int stackWitnessValue;
  int resumeCount;
  bool finished;
};

std::vector<CthThread> threads;
std::vector<CmiIsomallocContext> contexts;
std::vector<ThreadState *> states;

unsigned char expectedByte(int index, int i) {
  return (unsigned char)((index * 37 + i * 11 + 5) & 0xff);
}

void threadBody(void *arg) {
  ThreadState *st = (ThreadState *)arg;
  CthThread me = CthSelf();

  /* Everything from here to the matching Push runs with this thread's heap
     installed, which is what puts the allocation below inside the context. */
  CthInterceptionsDeactivatePop(me);

  int onStack = 0xC0FFEE + st->index;
  st->stackWitness = &onStack;
  st->stackWitnessValue = onStack;

  st->payload = (unsigned char *)malloc(kPayload);
  if (st->payload == nullptr)
    CkAbort("migthread: allocation inside the thread heap failed\n");
  if (!CmiIsomallocInRange(st->payload))
    CkAbort("migthread: thread heap allocation landed outside Isomalloc\n");
  for (int i = 0; i < kPayload; i++)
    st->payload[i] = expectedByte(st->index, i);

  CthInterceptionsDeactivatePush(me);
  CthSuspend();
  CthInterceptionsDeactivatePop(me);

  /* Resumed on the far side of a pack/unpack. */
  st->resumeCount++;

  if (st->stackWitness != &onStack)
    CkAbort("migthread: rank %d stack moved across the round trip (%p -> %p)\n",
            st->index, (void *)st->stackWitness, (void *)&onStack);
  if (onStack != st->stackWitnessValue)
    CkAbort("migthread: rank %d stack contents changed across the round trip\n",
            st->index);

  for (int i = 0; i < kPayload; i++) {
    if (st->payload[i] != expectedByte(st->index, i))
      CkAbort("migthread: rank %d heap byte %d changed across the round trip\n",
              st->index, i);
  }

  st->finished = true;
  CthInterceptionsDeactivatePush(me);
  CthSuspend();
}

/* Pack a thread out and unpack it back in, releasing the packed-from copy the
   way a real migration would. */
CthThread roundTrip(CthThread t) {
  PUP::sizer sizer(PUP::er::IS_MIGRATION);
  CthPup((pup_er)&sizer, t);
  const size_t len = sizer.size();

  std::vector<char> buf(len);

  PUP::toMem out(buf.data(), PUP::er::IS_MIGRATION);
  out.becomeDeleting();
  CthThread packed = CthPup((pup_er)&out, t);
  if (packed != t)
    CkAbort("migthread: packing returned a different thread\n");

  PUP::fromMem in(buf.data(), PUP::er::IS_MIGRATION);
  CthThread revived = CthPup((pup_er)&in, nullptr);
  if (revived == nullptr)
    CkAbort("migthread: unpacking produced no thread\n");
  return revived;
}

} // namespace

class Main : public CBase_Main {
  int pending;

public:
  Main(CkArgMsg *m) {
    delete m;
    mainProxy = thisProxy;

    if (!CthMigratable())
      CkAbort("migthread: migratable threads are unavailable -- was Isomalloc "
              "disabled?\n");
    /* TCharm gates several decisions on this, so a memory module that failed
       to announce itself would misbehave in ways far from here. */
    if (!CmiMemoryIs(CMI_MEMORY_IS_ISOMALLOC))
      CkAbort("migthread: the Isomalloc memory module is not installed -- link "
              "with `charmc -memory isomalloc`\n");

    CkPrintf("migthread: creating %d migratable threads\n", kNumThreads);

    /* One context per thread, partitioned the way TCharm partitions by array
       index: the split is by thread, not by PE, so it does not depend on how
       many PEs the job has. */
    for (int i = 0; i < kNumThreads; i++) {
      CmiIsomallocContext ctx = CmiIsomallocContextCreate(i, kNumThreads + 1);
      if (ctx.opaque == nullptr)
        CkAbort("migthread: could not create an Isomalloc context\n");
      CmiIsomallocContextEnableRandomAccess(ctx);

      ThreadState *st = new ThreadState();
      st->index = i;
      st->payload = nullptr;
      st->stackWitness = nullptr;
      st->stackWitnessValue = 0;
      st->resumeCount = 0;
      st->finished = false;

      CthThread t = CthCreateMigratable(threadBody, st, 0, ctx);
      CthSetStrategyDefault(t);

      contexts.push_back(ctx);
      states.push_back(st);
      threads.push_back(t);
    }

    pending = kNumThreads;
    for (int i = 0; i < kNumThreads; i++)
      CthAwaken(threads[i]);

    /* Let each thread run up to its first suspend, then round-trip it. The
       check runs from a scheduled callback so the threads have actually had
       the processor. */
    CcdCallFnAfter(
        [](void *obj, double) { ((Main *)obj)->afterFirstRun(); }, this, 1.0);
  }

  void afterFirstRun() {
    for (int i = 0; i < kNumThreads; i++) {
      if (states[i]->payload == nullptr)
        CkAbort("migthread: thread %d never ran\n", i);
      threads[i] = roundTrip(threads[i]);
      CthSetStrategyDefault(threads[i]);
    }
    CkPrintf("migthread: %d threads packed and unpacked; resuming\n",
             kNumThreads);

    for (int i = 0; i < kNumThreads; i++)
      CthAwaken(threads[i]);

    CcdCallFnAfter([](void *obj, double) { ((Main *)obj)->afterResume(); }, this,
                   1.0);
  }

  void afterResume() {
    for (int i = 0; i < kNumThreads; i++) {
      if (!states[i]->finished)
        CkAbort("migthread: thread %d did not resume after the round trip\n", i);
      if (states[i]->resumeCount != 1)
        CkAbort("migthread: thread %d resumed %d times, expected 1\n", i,
                states[i]->resumeCount);
    }
    CkPrintf("migthread: all %d threads survived the round trip\n", kNumThreads);
    CkExit();
  }

  void done() { CkExit(); }
};

#include "migthread.def.h"

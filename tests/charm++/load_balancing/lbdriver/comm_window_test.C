// Exercise the real LBDatabase send paths without starting Charm/LCI/CUDA.
// Expose fixture storage only after loading the dependency headers.
#include "lbdb.h"
#include "LBObj.h"
#include "LBOM.h"
#include "LBComm.h"
#include "LBMachineUtil.h"
#include <vector>
#include <unordered_map>
#define private public
#include "LBDatabase.h"
#undef private
#include "ck.h"
#include <cassert>
#include <cstdio>

extern void lbsimParseArgs(int&, char**);
CkpvDeclare(bool, CkSyncBarrierInited);
CkGroupID _syncBarrier;
void* CkLocalBranch(CkGroupID) { std::abort(); }

// Only the active record's handle is used by LBDatabase::Send.
static LDObjHandle fixtureHandle;
CkLocRec::CkLocRec(CkLocMgr*, bool, bool, const CkArrayIndex&, CmiUInt8)
    : ldHandle(fixtureHandle) {}
CkLocRec::~CkLocRec() {}
static CkLocRec* active = nullptr;
CkLocRec* CkActiveLocRec() { return active; }

static void expect(LBDatabase& db, int messages, int bytes) {
  std::vector<LDCommData> records(db.GetCommDataSz());
  db.GetCommData(records.data());
  int actualMessages = 0, actualBytes = 0;
  for (const auto& r : records) {
    actualMessages += r.messages;
    actualBytes += r.bytes;
  }
  assert(actualMessages == messages && actualBytes == bytes);
}

int main(int argc, char** argv) {
  lbsimParseArgs(argc, argv);
  bool barrierInited = false;
  CMK_TAG(Cpv_, CkSyncBarrierInited) = &barrierInited;
  LBDatabase db;
  db.statsAreOn = true;
  _lb_args.traceComm() = true;
  LDOMHandle om{};
  om.id.id.idx = 1;
  fixtureHandle.omhandle = om;
  fixtureHandle.id = 10;
  fixtureHandle.handle = 0;
  LBObj first(fixtureHandle);
  db.objs.emplace_back(&first);
  CkLocRec rec1(nullptr, false, false, CkArrayIndex(), 10);
  fixtureHandle.id = 11;
  fixtureHandle.handle = 1;
  LBObj second(fixtureHandle);
  db.objs.emplace_back(&second);
  CkLocRec rec2(nullptr, false, false, CkArrayIndex(), 11);
  CmiUInt8 dest = 20, dests[] = {20, 21};

  active = &rec1;
  db.Send(om, dest, 100, 1);
  expect(db, 1, 100);
  first.setJoinedStep(true);
  first.IncrementTime(1.0, 1.0);
  first.IncrementGPUTime(1.0);
  db.Send(om, dest, 100, 1);
  db.Send(om, dest, 100, 1, 1); // array recordSend uses force=1
  db.MulticastSend(om, dests, 2, 100, 2);
  expect(db, 1, 100);
  LBRealType wall, cpu, gpu;
  first.getTime(&wall, &cpu);
  first.getGPUTime(&gpu);
  assert(wall == 0.0 && gpu == 0.0);

  active = &rec2; // another object on this PE has not joined
  db.Send(om, dest, 100, 1, 1);
  db.MulticastSend(om, dests, 2, 200, 2);
  expect(db, 4, 400);

  active = &rec1;
  first.setJoinedStep(false); // AtSyncWait resumes measurement
  first.IncrementTime(1.0, 1.0);
  first.IncrementGPUTime(1.0);
  db.Send(om, dest, 100, 1, 1);
  db.MulticastSend(om, dests, 2, 200, 2);
  expect(db, 7, 700);
  first.getTime(&wall, &cpu);
  first.getGPUTime(&gpu);
  assert(wall == 1.0 && gpu == 1.0);

  db.Send(om, first.GetLDObjHandle().id, 100, 0, 1); // self messages stay excluded
  expect(db, 7, 700);
  db.statsAreOn = false;
  db.Send(om, dest, 100, 1);
  db.MulticastSend(om, dests, 2, 200, 2);
  expect(db, 7, 700);
  db.Send(om, dest, 100, 1, 1); // preserve explicit force for an open object
  expect(db, 8, 800);
  first.setJoinedStep(true);
  db.Send(om, dest, 100, 1, 1);
  expect(db, 8, 800);
  active = nullptr;
  db.Send(om, dest, 100, 1, 1); // preserve processor-origin forced records
  expect(db, 9, 900);
  db.objs.clear(); // fixture objects have stack ownership
  delete db.commTable;
  std::puts("comm measurement window: PASS");
}

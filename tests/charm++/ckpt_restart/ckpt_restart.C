// Disk checkpoint and restart, also with the number of PEs and the number of
// processes changing between the checkpoint and the restart.
//
// The program runs -steps N steps of a 48-element ring (a bounded 1D array):
// every element exchanges its value with both neighbours and updates it with a
// fixed formula, and every 4 steps it calls AtSync, so with RotateLB or GreedyLB
// the elements are away from their home PEs when the checkpoint is taken. The
// ring's final values do not depend on the PE or process count, so a run that is
// checkpointed and restarted must end with the summary of an uninterrupted run
// (-write-check F writes it, -expect-check F compares with it).
//
// What is checked, and the old test each check comes from (chkpt = the former
// tests/charm++/chkpt, unbounded_restart = the former tests/charm++/unbounded_restart):
// - chkpt: the [migratable] mainchare is restored: its migration constructor gets
//   the restart's CkArgMsg, and its pup then overwrites the values that constructor
//   set (a, b[2]); its step count is the checkpoint's.
// - chkpt: the 1D array's data is recovered (here: every ring value, bit for bit,
//   and the final checksum), with GreedyLB (+LBDebug) or RotateLB, on fewer and
//   more PEs; every step's reduction carries the step number and all 48 elements.
// - chkpt: the plain chare is restored, with its data, when the PE count is the
//   same, and only then (_chareRestored).
// - chkpt: group and nodegroup data are restored per PE / per node when the PE /
//   node count is the same; otherwise every PE / node gets PE 0's / node 0's copy.
//   New: a tag set on every branch before the checkpoint must be on every branch.
// - New: readonlies are restored (roSteps, roRingSize, checked on every PE).
// - unbounded_restart: the ids of an unbounded, dynamically inserted array survive
//   the restart; every element is reachable by index afterwards and replies with
//   its recorded id.
// - unbounded_restart: after a restart 16 new elements are inserted; their unique
//   numbers come from the allocator region [2^(U-1), 2^U) and every id is distinct.
// - unbounded_restart: -c makes a restarted run checkpoint again, for chains of
//   restarts; make testchain runs nine, and each restart may cost the allocator's
//   pool only what its insertions use.
// - New: elements destroyed before the checkpoint do not exist after the restart
//   (a broadcast census returns exactly the live set, with matching ids).
// - New: the checkpoint is written by 1 or 2 processes and restarted in 2 or 1.
//
// Plain chare and insertions: an array element inserted on a PE that also hosts a
// plain (non-array) chare can take that chare's slot in the PE's chare table, and
// when the element is later destroyed or migrates away the chare's next message
// crashes (runtime defect; an issue is being filed). So the plain chare lives on
// PE 0, elements are never inserted on PE 0, and the chare is messaged only
// before the insertions of the run. PE 0 always hosts the mainchare, whose slot an
// element inserted there would take as well, so the program needs at least 2 PEs
// until the defect is fixed (at 1 PE a checked build aborts in ~Chare).
//
// Options: -steps N (default 16); -ckpt-at K: checkpoint to log/ after step K
// (without it the 16 insertions of step K happen after step 6); -c;
// -write-check F; -expect-check F.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "pup_stl.h"
#include "ckpt_restart.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CProxy_Ring ringProxy;
/*readonly*/ CProxy_UbElem ubProxy;
/*readonly*/ CProxy_PlainChare chareProxy;
/*readonly*/ CProxy_PeGroup groupProxy;
/*readonly*/ CProxy_NodeData nodeGroupProxy;
/*readonly*/ int roSteps;
/*readonly*/ int roRingSize;

static const int ringSize = 48;
static const int lbPeriod = 4;      // AtSync at every step that is a multiple of this
static const int nBase = 24;        // unbounded elements inserted at the start
static const int nNew = 16;         // inserted at step K, or after a restart
static const int newBaseStep = 100;  // generation g inserts [100 g, 100 g + 16)
static const int chareValue = 7;

void unbufferStdout(void) { setvbuf(stdout, NULL, _IONBF, 0); }

// Unbounded elements avoid PE 0, where the plain chare lives (see the top).
static int insertPe(int i) { return CkNumPes() > 1 ? 1 + i % (CkNumPes() - 1) : 0; }
static bool baseDestroyed(int i) { return i % 4 == 3; }

struct IdPair
{
  int idx;
  int pad;
  CmiUInt8 id;
};

class Main : public CBase_Main
{
  // Pupped state.
  int nSteps = 16, ckptAt = -1, phaseStep = 6;
  int step = 0;
  int gen = 0;              // restarts so far
  int ckptPes = 0, ckptNodes = 0;
  bool chareAlive = true;   // false once a restart on another PE count dropped it
  int a = 0, b[2] = {0, 0};  // chkpt's mainchare data
  std::vector<double> vals;           // ring values after the last step
  std::map<int, CmiUInt8> live;       // unbounded: index -> id
  std::map<int, CmiUInt8> dead;       // destroyed before the checkpoint
  CmiUInt8 lastMin = 0;

  // Not pupped: this run's progress.
  bool migCtorGotArgs = false;
  bool pendingCkpt = false;
  bool reckptDone = false;
  bool chain = false;
  std::string writeCheck, expectCheck;
  int waitCount = 0;
  enum Phase { BASE_INSERT, BASE_DESTROY, BASE_CENSUS, NEW_INSERT, NEW_PINGS,
               NEW_CENSUS, FINAL_CENSUS, RESTART_CENSUS } phase = BASE_INSERT;

  void parseRunArgs(char** argv)
  {
    chain = CmiGetArgFlagDesc(argv, "-c", "checkpoint again after a restart") != 0;
    char* s = nullptr;
    if (CmiGetArgStringDesc(argv, "-write-check", &s, "write the summary")) writeCheck = s;
    s = nullptr;
    if (CmiGetArgStringDesc(argv, "-expect-check", &s, "compare the summary")) expectCheck = s;
  }
  int newBase() const { return newBaseStep * (gen > 0 ? gen : 1); }

public:
  Main(CkArgMsg* m)
  {
    char** argv = m->argv;
    CmiGetArgIntDesc(argv, "-steps", &nSteps, "number of steps");
    if (CmiGetArgIntDesc(argv, "-ckpt-at", &ckptAt, "checkpoint after this step"))
      phaseStep = ckptAt;
    parseRunArgs(argv);
    CkEnforce(phaseStep >= 1 && phaseStep < nSteps);
    if (CkNumPes() < 2)
      CkAbort("ckpt_restart needs at least 2 PEs: elements must not be inserted on PE 0"
              " (see the top of ckpt_restart.C)\n");
    delete m;
    a = 123; b[0] = 456; b[1] = 789;
    roSteps = nSteps;
    roRingSize = ringSize;
    mainProxy = thisProxy;
    vals.assign(ringSize, 0.0);
    ringProxy = CProxy_Ring::ckNew(ringSize);
    ubProxy = CProxy_UbElem::ckNew(CkArrayOptions());  // no bounds: hashed ids
    groupProxy = CProxy_PeGroup::ckNew();
    nodeGroupProxy = CProxy_NodeData::ckNew();
    chareProxy = CProxy_PlainChare::ckNew(0);
    CkPrintf("ckpt_restart: %d steps, %d PEs, %d processes, checkpoint after step %d\n",
             nSteps, CkNumPes(), CkNumNodes(), ckptAt);
    // The plain chare is messaged before any insertion (see the top).
    chareProxy.sayHi(chareValue);
  }

  Main(CkMigrateMessage* m) : CBase_Main(m)
  {
    mainProxy = thisProxy;
    // The restart hands the mainchare's migration constructor its CkArgMsg.
    migCtorGotArgs = m != nullptr && ((CkArgMsg*)m)->argc >= 1;
    parseRunArgs(CkGetArgv());
    a = 987; b[0] = 654; b[1] = 321;  // pup must overwrite these
  }

  void pup(PUP::er& p)
  {
    p | nSteps; p | ckptAt; p | phaseStep; p | step; p | gen;
    p | ckptPes; p | ckptNodes; p | chareAlive;
    p | a; p(b, 2);
    p | vals; p | live; p | dead; p | lastMin;
  }

  void chareAck(int value, int pe)
  {
    CkEnforce(pe == 0);
    if (value != chareValue) CkAbort("plain chare has %d, expected %d\n", value, chareValue);
    if (!_restarted)
    {
      phase = BASE_INSERT;
      waitCount = nBase;
      for (int i = 0; i < nBase; i++) ubProxy[i].insert(insertPe(i));
      ubProxy.doneInserting();
    }
    else
      insertNew();
  }

  void reportId(int idx, CmiUInt8 id)
  {
    CkEnforceMsg(live.find(idx) == live.end() && dead.find(idx) == dead.end(),
                 "element constructed twice");
    live[idx] = id;
    if (--waitCount) return;
    if (phase == BASE_INSERT)
    {
      phase = BASE_DESTROY;
      for (int i = 0; i < nBase; i++)
        if (baseDestroyed(i)) { waitCount++; ubProxy[i].die(); }
    }
    else
    {
      CkEnforce(phase == NEW_INSERT);
      checkNewIds();
    }
  }

  void destroyed(int idx)
  {
    auto it = live.find(idx);
    CkEnforce(it != live.end());
    dead[idx] = it->second;
    live.erase(it);
    if (--waitCount) return;
    phase = BASE_CENSUS;
    startCensus();
  }

  void startCensus()
  {
    ubProxy.census();
  }

  // The broadcast reaches exactly the live elements, each with its recorded id.
  void census(CkReductionMsg* m)
  {
    std::map<int, CmiUInt8> got;
    auto* e = (CkReduction::setElement*)m->getData();
    while (e && e->dataSize > 0)
    {
      IdPair pr;
      memcpy(&pr, e->data, sizeof(pr));
      CkEnforceMsg(got.emplace(pr.idx, pr.id).second, "census: an element replied twice");
      e = e->next();
    }
    delete m;
    for (auto& d : dead)
      if (got.count(d.first))
        CkAbort("census: element %d was destroyed before the checkpoint but exists\n",
                d.first);
    if (got != live)
      CkAbort("census: %d elements replied, %d expected (phase %d)\n", (int)got.size(),
              (int)live.size(), (int)phase);
    switch (phase)
    {
      case BASE_CENSUS:
        CkPrintf("Census: %d unbounded elements live, %d destroyed\n", (int)live.size(),
                 (int)dead.size());
        nextStep();
        break;
      case RESTART_CENSUS:
        CkPrintf("Census after restart: the %d live elements kept their ids, the %d"
                 " destroyed ones are gone\n",
                 (int)live.size(), (int)dead.size());
        // Restored chare: message it now, before this run's insertions.
        if (chareAlive && CkNumPes() == ckptPes)
          chareProxy.check(chareValue);
        else
        {
          chareAlive = false;
          CkPrintf("Plain chare not restored (%d PEs at the checkpoint, %d now)\n",
                   ckptPes, CkNumPes());
          insertNew();
        }
        break;
      case NEW_CENSUS:
        afterNew();
        break;
      case FINAL_CENSUS:
        finish();
        break;
      default:
        CkAbort("census in phase %d\n", (int)phase);
    }
  }

  void insertNew()
  {
    phase = NEW_INSERT;
    waitCount = nNew;
    CkPrintf("Step %d: inserting %d new elements at %d\n", step, nNew, newBase());
    for (int i = newBase(); i < newBase() + nNew; i++) ubProxy[i].insert(insertPe(i));
    ubProxy.doneInserting();
  }

  void checkNewIds()
  {
    const int U = ck::objid::getLayout().uniqueBits;
    const CmiUInt8 mask = ((CmiUInt8)1 << U) - 1;
    const CmiUInt8 half = (CmiUInt8)1 << (U - 1);
    std::set<CmiUInt8> seen;
    CmiUInt8 minNew = ~(CmiUInt8)0;
    for (auto* mp : {&live, &dead})
      for (auto& e : *mp)
        CkEnforceMsg(seen.insert(e.second).second, "two elements share an id");
    if (gen > 0)
    {
      for (auto& e : live)
      {
        if (e.first < newBase()) continue;
        if ((e.second & mask) < half)
          CkAbort("restart: new element %d has unique part %llu below 2^%d: not from"
                  " the allocator\n",
                  e.first, (unsigned long long)(e.second & mask), U - 1);
        if ((e.second & mask) < minNew) minNew = e.second & mask;
      }
      if (gen >= 2)
        CkPrintf("Generation %d: the previous restart consumed %llu of the pool's %llu"
                 " (lowest new number %llu, was %llu)\n",
                 gen, (unsigned long long)(minNew - lastMin), (unsigned long long)half,
                 (unsigned long long)minNew, (unsigned long long)lastMin);
      lastMin = minNew;
      CkPrintf("The %d new ids came from the allocator\n", nNew);
    }
    CkPrintf("All %d ids distinct; pinging every live element\n", (int)seen.size());
    phase = NEW_PINGS;
    waitCount = live.size();
    for (auto& e : live) ubProxy[e.first].ping();
  }

  void pong(int idx, CmiUInt8 id)
  {
    auto it = live.find(idx);
    CkEnforce(it != live.end());
    if (it->second != id)
      CkAbort("element %d: id %llx, recorded %llx\n", idx, (unsigned long long)id,
              (unsigned long long)it->second);
    if (--waitCount) return;
    phase = NEW_CENSUS;
    startCensus();
  }

  void afterNew()
  {
    const int expectLive = nBase - (int)dead.size() + nNew * (gen > 0 ? gen : 1);
    if ((int)live.size() != expectLive)
      CkAbort("%d live elements, expected %d\n", (int)live.size(), expectLive);
    if (_restarted && chain && !reckptDone)
    {
      reckptDone = true;
      CkPrintf("Checkpointing again (generation %d)\n", gen);
      startCheckpoint();
      return;
    }
    nextStep();
  }

  void startCheckpoint()
  {
    ckptPes = CkNumPes();
    ckptNodes = CkNumNodes();
    pendingCkpt = true;
    CkStartCheckpoint("log", CkCallback(CkIndex_Main::resumed(), thisProxy));
  }

  // The checkpoint's callback: after this run's checkpoint, and as the first
  // message of a restarted run.
  void resumed()
  {
    if (pendingCkpt)
    {
      pendingCkpt = false;
      CkPrintf("Checkpoint done at step %d\n", step);
      if (_restarted) nextStep();  // a -c checkpoint: its insertions are done
      else checkGroups();
      return;
    }
    CkEnforce(_restarted);
    gen++;
    CkPrintf("Restarted (generation %d) at step %d on %d PEs, %d processes; checkpoint"
             " written on %d PEs, %d processes\n",
             gen, step, CkNumPes(), CkNumNodes(), ckptPes, ckptNodes);
    CkEnforceMsg(migCtorGotArgs, "the mainchare's migration constructor got no CkArgMsg");
    if (a != 123 || b[0] != 456 || b[1] != 789)
      CkAbort("mainchare data not restored: a=%d b={%d,%d}\n", a, b[0], b[1]);
    CkEnforce(step == phaseStep && _oldNumPes == ckptPes);
    if (_chareRestored != (chareAlive && CkNumPes() == ckptPes))
      CkAbort("_chareRestored=%d, but the checkpoint had %d PEs and this run %d\n",
              (int)_chareRestored, ckptPes, CkNumPes());
    // The restored ring values, bit for bit.
    ringProxy.report();
  }

  void ringReported(CkReductionMsg* m)
  {
    const double* v = (const double*)m->getData();
    CkEnforce(m->getSize() == (int)(sizeof(double) * (ringSize + 3)));
    for (int i = 0; i < ringSize; i++)
      if (v[i] != vals[i])
        CkAbort("ring element %d restored as %.17g, was %.17g\n", i, v[i], vals[i]);
    CkEnforce(v[ringSize + 1] == ringSize);
    delete m;
    checkGroups();
  }

  // Group and nodegroup branches (and the readonlies, on every PE).
  void checkGroups()
  {
    const bool otherPes = _restarted && CkNumPes() != ckptPes;
    const bool otherNodes = _restarted && CkNumNodes() != ckptNodes;
    waitCount = 2;
    groupProxy.check(nSteps, phaseStep, otherPes);
    nodeGroupProxy.check(nSteps, phaseStep, otherNodes);
  }

  void groupsChecked(CkReductionMsg* m)
  {
    delete m;
    if (--waitCount) return;
    CkPrintf("Group and nodegroup data checked on every PE and node\n");
    if (_restarted)
    {
      phase = RESTART_CENSUS;
      startCensus();  // then the plain chare, then the insertions
    }
    else
      insertNew();
  }

  void nextStep()
  {
    if (step == nSteps)
    {
      phase = FINAL_CENSUS;
      startCensus();
      return;
    }
    step++;
    ringProxy.step(step);
  }

  void stepDone(CkReductionMsg* m)
  {
    const double* v = (const double*)m->getData();
    CkEnforce(m->getSize() == (int)(sizeof(double) * (ringSize + 3)));
    if (v[ringSize] != (double)step * ringSize || v[ringSize + 1] != ringSize)
      CkAbort("step %d: reduction says step sum %g over %g elements\n", step,
              v[ringSize], v[ringSize + 1]);
    for (int i = 0; i < ringSize; i++) vals[i] = v[i];
    const int away = (int)v[ringSize + 2];
    delete m;
    if (step == phaseStep)
    {
      CkPrintf("Step %d: %d of the %d ring elements are away from their home PE\n", step,
               away, ringSize);
      waitCount = 2;
      groupProxy.setTag(phaseStep);
      nodeGroupProxy.setTag(phaseStep);
      return;
    }
    nextStep();
  }

  void tagged(CkReductionMsg* m)
  {
    delete m;
    if (--waitCount) return;
    if (ckptAt > 0)
    {
      CkPrintf("Checkpointing after step %d\n", step);
      startCheckpoint();
    }
    else
      checkGroups();
  }

  void finish()
  {
    double checksum = 0;
    for (int i = 0; i < ringSize; i++) checksum += (i + 1) * vals[i];
    int baseLive = 0, newLive = 0;
    for (auto& e : live)
    {
      if (e.first < nBase) baseLive++;
      else if (e.first >= newBase() && e.first < newBase() + nNew) newLive++;
    }
    char line[256];
    snprintf(line, sizeof(line),
             "steps %d ring %d checksum %.17g ubbase %d ubdestroyed %d ubnew %d"
             " idsdistinct 1\n",
             nSteps, ringSize, checksum, baseLive, (int)dead.size(), newLive);
    CkPrintf("Summary: %s", line);
    if (!writeCheck.empty())
    {
      FILE* f = fopen(writeCheck.c_str(), "w");
      CkEnforceMsg(f, "cannot open the -write-check file");
      fputs(line, f);
      fclose(f);
    }
    if (!expectCheck.empty())
    {
      FILE* f = fopen(expectCheck.c_str(), "r");
      if (!f) CkAbort("cannot open %s\n", expectCheck.c_str());
      char ref[256] = "";
      if (!fgets(ref, sizeof(ref), f)) ref[0] = 0;
      fclose(f);
      int s[2], r[2], c[3], d[3];
      double x[2];
      const char* fmt = "steps %d ring %d checksum %lg ubbase %d ubdestroyed %d ubnew %d";
      if (sscanf(line, fmt, &s[0], &r[0], &x[0], &c[0], &c[1], &c[2]) != 6 ||
          sscanf(ref, fmt, &s[1], &r[1], &x[1], &d[0], &d[1], &d[2]) != 6)
        CkAbort("cannot parse the summaries:\n  this run: %s  reference: %s", line, ref);
      if (s[0] != s[1] || r[0] != r[1] || c[0] != d[0] || c[1] != d[1] || c[2] != d[2] ||
          std::fabs(x[0] - x[1]) > 1e-12 * std::fabs(x[1]))
        CkAbort("summary mismatch:\n  this run:  %s  reference: %s", line, ref);
      CkPrintf("Summary matches %s\n", expectCheck.c_str());
    }
    CkPrintf("All done\n");
    CkExit();
  }
};

class Ring : public CBase_Ring
{
  double v;
  int cur = 0;      // last step started
  int got = 0;      // neighbour values received for step cur + (started ? 0 : 1)
  bool started = false;
  double left = 0, right = 0;

  // This element's value at its index, then the step, 1 (a count of elements)
  // and 1 if the element is away from its home PE.
  std::vector<double> values()
  {
    std::vector<double> out(ringSize + 3, 0.0);
    out[thisIndex] = v;
    out[ringSize] = cur;
    out[ringSize + 1] = 1;
    out[ringSize + 2] = thisProxy.ckLocMgr()->homePe(thisIndexMax) != CkMyPe();
    return out;
  }
  void contributeValues()
  {
    contribute(values(), CkReduction::sum_double,
               CkCallback(CkIndex_Main::stepDone(nullptr), mainProxy));
  }
  void tryUpdate()
  {
    if (!started || got < 2) return;
    v = 0.5 * v + 0.25 * (left + right) + 0.001 * ((thisIndex * 31 + cur * 17) % 23);
    started = false;
    got = 0;
    if (cur % lbPeriod == 0)
      AtSync();
    else
      contributeValues();
  }

public:
  Ring() : v(1.0 + 0.01 * thisIndex) { usesAtSync = true; }
  Ring(CkMigrateMessage* m) : CBase_Ring(m) {}
  void pup(PUP::er& p)
  {
    p | v; p | cur; p | got; p | started; p | left; p | right;
  }
  void step(int s)
  {
    CkEnforce(s == cur + 1 && !started);
    cur = s;
    started = true;
    thisProxy[(thisIndex + 1) % ringSize].recv(s, 1, v);
    thisProxy[(thisIndex + ringSize - 1) % ringSize].recv(s, 0, v);
    tryUpdate();
  }
  void recv(int s, int fromLeft, double x)
  {
    CkEnforce(s == (started ? cur : cur + 1));
    (fromLeft ? left : right) = x;
    got++;
    tryUpdate();
  }
  void ResumeFromSync() { contributeValues(); }
  void report()
  {
    contribute(values(), CkReduction::sum_double,
               CkCallback(CkIndex_Main::ringReported(nullptr), mainProxy));
  }
};

class UbElem : public CBase_UbElem
{
public:
  UbElem() { mainProxy.reportId(thisIndex, ckGetID().getElementID()); }
  UbElem(CkMigrateMessage* m) : CBase_UbElem(m) {}
  void ping() { mainProxy.pong(thisIndex, ckGetID().getElementID()); }
  void census()
  {
    IdPair pr{thisIndex, 0, ckGetID().getElementID()};
    contribute(sizeof(pr), &pr, CkReduction::set,
               CkCallback(CkIndex_Main::census(nullptr), mainProxy));
  }
  void die()
  {
    mainProxy.destroyed(thisIndex);
    ckDestroy();
  }
  void pup(PUP::er& p) { CBase_UbElem::pup(p); }
};

class PlainChare : public CBase_PlainChare
{
  int value = -1;

public:
  PlainChare() {}
  PlainChare(CkMigrateMessage* m) : CBase_PlainChare(m) {}
  void sayHi(int x)
  {
    value = x;
    mainProxy.chareAck(value, CkMyPe());
  }
  void check(int expected)
  {
    if (value != expected) CkAbort("plain chare restored with %d, expected %d\n", value, expected);
    CkPrintf("Plain chare restored on PE %d with its data\n", CkMyPe());
    mainProxy.chareAck(value, CkMyPe());
  }
  void pup(PUP::er& p) { p | value; }
};

class PeGroup : public CBase_PeGroup
{
  int pe;
  int tag = -1;

public:
  PeGroup() : pe(CkMyPe()) {}
  PeGroup(CkMigrateMessage* m) : CBase_PeGroup(m) {}
  void pup(PUP::er& p) { p | pe; p | tag; }
  void setTag(int t)
  {
    tag = t;
    contribute(CkCallback(CkIndex_Main::tagged(nullptr), mainProxy));
  }
  // fromPe0: a restart on another PE count gives every PE PE 0's copy.
  void check(int steps, int expectTag, int fromPe0)
  {
    if (roSteps != steps || roRingSize != ringSize)
      CkAbort("PE %d: readonlies roSteps=%d roRingSize=%d, expected %d and %d\n", CkMyPe(),
              roSteps, roRingSize, steps, ringSize);
    const int expectPe = fromPe0 ? 0 : CkMyPe();
    if (pe != expectPe || tag != expectTag)
      CkAbort("PE %d: group data pe=%d tag=%d, expected pe=%d tag=%d\n", CkMyPe(), pe, tag,
              expectPe, expectTag);
    pe = CkMyPe();  // so a later checkpoint holds this run's values
    contribute(CkCallback(CkIndex_Main::groupsChecked(nullptr), mainProxy));
  }
};

class NodeData : public CBase_NodeData
{
  int node;
  int tag = -1;

public:
  NodeData() : node(CkMyNode()) {}
  NodeData(CkMigrateMessage* m) : CBase_NodeData(m) {}
  void pup(PUP::er& p) { p | node; p | tag; }
  void setTag(int t)
  {
    tag = t;
    contribute(CkCallback(CkIndex_Main::tagged(nullptr), mainProxy));
  }
  void check(int steps, int expectTag, int fromNode0)
  {
    if (roSteps != steps) CkAbort("node %d: roSteps=%d, expected %d\n", CkMyNode(), roSteps, steps);
    const int expectNode = fromNode0 ? 0 : CkMyNode();
    if (node != expectNode || tag != expectTag)
      CkAbort("node %d: nodegroup data node=%d tag=%d, expected node=%d tag=%d\n",
              CkMyNode(), node, tag, expectNode, expectTag);
    node = CkMyNode();
    contribute(CkCallback(CkIndex_Main::groupsChecked(nullptr), mainProxy));
  }
};

#include "ckpt_restart.def.h"

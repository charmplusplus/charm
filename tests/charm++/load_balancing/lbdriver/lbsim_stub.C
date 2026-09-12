// The runtime, as much of it as lbsim's decision code touches, for the
// standalone build (lbsim.C, -DLBSIM_STANDALONE).
//
// lbsim runs DiffusionLB's own decision code -- the flow arithmetic, the
// selection metric, the selection loop, the cost model -- on virtual nodes
// inside one process. That code is written against the Charm++ headers and
// reaches the runtime for very little: the print and abort functions, a PE
// number, the physical-node queries the cost model's tier lookup makes, the
// pup size codec, the LB argument block, and the comm hash of LDStats. In
// the Charm++ build all of that comes from libck and libconverse, and with
// them comes LCI, whose startup calls cuInit and opens the NIC -- so the
// simulator, which never touches a GPU, cannot start on a login node.
//
// This file supplies those symbols instead, so the standalone lbsim links
// against nothing but METIS and the decision code and runs anywhere. Every
// definition here is either trivial (one PE, one node) or a copy of the
// runtime's own (the hash, the size codec), so the decisions are the same
// ones the chare makes. The functions are first declared by the headers
// included below; a later definition takes its linkage from that
// declaration, so nothing here needs to repeat extern "C".

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "BaseLB.h"
#include "DiffusionLoad.h"
#include "LBManager.h"

// ---- globals the module would define -------------------------------------

CkLBArgs _lb_args;

// The per-PE user-data layout LDObjData's constructor sizes its slot vector
// from (lbdb.h). LBManager declares and initialises it in the runtime; a
// per-PE variable is a thread-local pointer here, so it is pointed at one
// empty layout before anything constructs an LDObjData (lbsimParseArgs).
CkpvDeclare(LBUserDataLayout, lbobjdatalayout);

// DiffusionLB.C defines these for the chare; here the simulator sets them.
int diffusionLoadDimDevice = -1;
int diffusionNodeDeviceBound = 0;
int diffusionPpn = 1;

// ---- printing and aborting -----------------------------------------------

int CmiPrintf(const char* format, ...)
{
  va_list ap;
  va_start(ap, format);
  const int r = vprintf(format, ap);
  va_end(ap);
  fflush(stdout);
  return r;
}

void CmiAbort(const char* format, ...)
{
  va_list ap;
  va_start(ap, format);
  vfprintf(stderr, format, ap);
  va_end(ap);
  fputc('\n', stderr);
  fflush(stderr);
  exit(1);
}

// ---- one PE, one node ----------------------------------------------------

int CmiMyPe() { return 0; }
int CmiNodeOf(int pe) { return pe; }
int CmiPhysicalNodeID(int) { return 0; }
int CmiPhysicalRank(int pe) { return pe; }
int CmiPeOnSamePhysicalNode(int, int) { return 1; }
int CmiNumPesOnPhysicalNode(int) { return 1; }

// ---- pup size codec (src/util/pup_c.C) -----------------------------------

#define SIZE_APPROX_BITS 13

CMK_TYPEDEF_UINT2 pup_encodeSize(size_t s)
{
  CmiUInt2 power = 0;
  while (s > (1UL << SIZE_APPROX_BITS) - 1)
  {
    power++;
    if (s & (1UL << 6)) s += (1UL << 7);
    s >>= 8;
  }
  return (power << SIZE_APPROX_BITS) | s;
}

size_t pup_decodeSize(CMK_TYPEDEF_UINT2 a)
{
  const CmiUInt2 power = a >> SIZE_APPROX_BITS;
  const size_t factor = 1UL << (8 * power);
  const size_t base = a & ((1UL << SIZE_APPROX_BITS) - 1);
  return base * factor;
}

// ---- LDStats: the constructor and the comm hash (src/ck-ldb/BaseLB.C) -----

BaseLB::LDStats::LDStats(int npes, int complete)
    : n_migrateobjs(0), complete_flag(complete)
{
  procs.resize(npes);
}

static const unsigned int doublingPrimes[] = {
    3u,        7u,        17u,       37u,        73u,        157u,       307u,
    617u,      1217u,     2417u,     4817u,      9677u,      20117u,     40177u,
    80177u,    160117u,   320107u,   640007u,    1280107u,   2560171u,   5120173u,
    10240201u, 20480197u, 40960223u, 81920327u,  163840259u, 327680281u, 655360271u,
    1310720281u, 2621440291u, 4200000071u};

static unsigned int primeLargerThan(unsigned int x)
{
  int i = 0;
  while (doublingPrimes[i] <= x) i++;
  return doublingPrimes[i];
}

inline static int ObjKey(const CmiUInt8& oid, const int hashSize)
{
  return (int)(oid % hashSize);
}

void BaseLB::LDStats::makeCommHash()
{
  if (!objHash.empty()) return;
  hashSize = primeLargerThan(objData.size() * 2);
  objHash.assign(hashSize, -1);
  int i = 0;
  for (const auto& obj : objData)
  {
    const CmiUInt8& oid = obj.objID();
    int hash = ObjKey(oid, hashSize);
    while (objHash[hash] != -1) hash = (hash + 1) % hashSize;
    objHash[hash] = i++;
  }
}

void BaseLB::LDStats::deleteCommHash()
{
  objHash.clear();
  for (auto& comm : commData) comm.clearHash();
}

int BaseLB::LDStats::getHash(const CmiUInt8& oid, const LDOMid& mid)
{
  if (hashSize <= 0) return -1;
  const int hash = ObjKey(oid, hashSize);
  for (int id = 0; id < hashSize; id++)
  {
    const int index = (id + hash) % hashSize;
    if (index == -1 || objHash[index] == -1) return -1;
    if (objData[objHash[index]].objID() == oid && objData[objHash[index]].omID() == mid)
      return objHash[index];
  }
  return -1;
}

int BaseLB::LDStats::getHash(const LDObjKey& objKey)
{
  return getHash(objKey.objID(), objKey.omID());
}

// ---- the +LB flags ---------------------------------------------------------
//
// In the Charm++ build the runtime strips these from argv and LBManager
// parses them into _lb_args before main runs. Here lbsim does both, for the
// flags its header documents, and leaves the positional arguments in place.

void lbsimParseArgs(int& argc, char** argv)
{
  static LBUserDataLayout layout;
  CMK_TAG(Cpv_, lbobjdatalayout) = &layout;

  int out = 1;
  for (int i = 1; i < argc; i++)
  {
    const std::string a = argv[i];
    const bool hasNext = (i + 1 < argc);
    if (a == "+LBDiffusionCommOn") _lb_args.diffusionCommOn() = true;
    else if (a == "+LBDiffusionGpuDim") _lb_args.diffusionGpuDim() = true;
    else if (a == "+LBDiffusionHostDim") _lb_args.diffusionHostDim() = true;
    else if (a == "+LBnoMST") _lb_args.noMST() = true;
    else if (a == "+LBDiffusionNumNbors" && hasNext) _lb_args.diffusionNumNbors() = atoi(argv[++i]);
    else if (a == "+LBDebug" && hasNext) _lb_args.debug() = atoi(argv[++i]);
    else if (a == "+LBDiffusionBeta" && hasNext) _lb_args.diffusionBeta() = atof(argv[++i]);
    else if (a == "+LBDiffusionMinImbalance" && hasNext) _lb_args.diffusionMinImbalance() = atof(argv[++i]);
    else if (a == "+LBDiffusionMaxMoveFrac" && hasNext) _lb_args.diffusionMaxMoveFrac() = atof(argv[++i]);
    else if (a == "+LBLoadVectorAbove" && hasNext) _lb_args.loadVectorAbove() = atof(argv[++i]);
    else if (a == "+LBCostConfig" && hasNext) _lb_args.costConfig() = strdup(argv[++i]);
    else if (a.rfind("+p", 0) == 0) { /* one PE regardless */ }
    else if (!a.empty() && a[0] == '+')
    {
      fprintf(stderr, "lbsim: unknown option %s\n", a.c_str());
      exit(1);
    }
    else argv[out++] = argv[i];
  }
  argc = out;
}

// ---- the cost model --------------------------------------------------------
//
// DiffusionCostModel.C is written to be textually included after its header
// (DiffusionLB.C does the same), so it is pulled in here rather than compiled
// on its own.
#include "DiffusionCostModel.h"
#include "DiffusionCostModel.C"

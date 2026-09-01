/** \file cklocation.C
 *  \addtogroup CkArrayImpl
 *
 *  The location manager keeps track of an indexed set of migratable objects.
 *  It is used by the array manager to locate array elements, interact with the
 *  load balancer, and perform migrations.
 *
 *  Orion Sky Lawlor, olawlor@acm.org 9/29/2001
 */

#include "TopoManager.h"
#include "charm++.h"
#include "ck.h"
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unistd.h>
#include "cksyncbarrier.h"
#include "hilbert.h"
#include "partitioning_strategies.h"
#include "pup_stl.h"
#include "register.h"
#include "trace.h"
#include <algorithm>
#include <limits>
#include <sstream>
#include <stdarg.h>
#include <vector>

#if CMK_CUDA || CMK_HIP

#include "hapi.h"
#include "gpumanager.h"

CsvExtern(GPUManager, gpu_manager);

#endif // CMK_CUDA || CMK_HIP

#if CMK_LBDB_ON
#  include "LBManager.h"
#  include "MetaBalancer.h"
#  if CMK_GLOBAL_LOCATION_UPDATE
#    include "BaseLB.h"
#    include "init.h"
#  endif
CkpvExtern(int, _lb_obj_index);  // for lbdb user data for obj index
#endif                           // CMK_LBDB_ON

CpvExtern(std::vector<NcpyOperationInfo*>, newZCPupGets);  // used for ZC Pup
#ifndef CMK_CHARE_USE_PTR
CkpvExtern(int, currentChareIdx);
#endif

#if CMK_GRID_QUEUE_AVAILABLE
CpvExtern(void*, CkGridObject);
#endif

// CHARM_DEBUG_MIGRATE: trace the steps of the export-and-pull device migration
// handshake and the in-flight immigration registry, so a stall can be located to
// a specific step rather than inferred.
static inline bool migDbg() {
  static const bool on = (getenv("CHARM_DEBUG_MIGRATE") != nullptr);
  return on;
}

// Process-wide table of which PE inside this process currently owns each
// element, keyed by the globally unique (array id, element id) pair.
//
// The per-PE location cache cannot answer this question reliably. It names a PE
// from whatever it last heard, so it can still claim an element is in this
// process after the element left -- if it left from a *different* PE of the
// process, this PE has heard nothing. Every arrival and departure inside the
// process runs through CkLocMgr::insertRec and removeFromTable, so this table
// cannot be stale about residency.
//
// That is exactly what a device zerocopy sender needs: "same process" is "same
// GPU", and it decides whether the receiver may read the sender's device
// pointer directly (MEMCPY) or the payload has to be staged. Getting it wrong
// in the optimistic direction hands the receiver a pointer into an address
// space it cannot read. A function-local static rather than a Csv: one copy per
// process image is precisely the scope wanted, and C++11 guarantees the
// initialization is thread safe without any runtime plumbing.
namespace {
struct CkResidentTable {
  std::mutex lock;
  std::unordered_map<CmiUInt8, int> owner;
};
CkResidentTable& residentTable() {
  static CkResidentTable table;
  return table;
}
}  // namespace

#define ARRAY_DEBUG_OUTPUT 0

#if ARRAY_DEBUG_OUTPUT
#  define DEB(x) CkPrintf x   // General debug messages
#  define DEBI(x) CkPrintf x  // Index debug messages
#  define DEBC(x) CkPrintf x  // Construction debug messages
#  define DEBS(x) CkPrintf x  // Send/recv/broadcast debug messages
#  define DEBM(x) CkPrintf x  // Migration debug messages
#  define DEBL(x) CkPrintf x  // Load balancing debug messages
#  define DEBN(x) CkPrintf x  // Location debug messages
#  define DEBB(x) CkPrintf x  // Broadcast debug messages
#  define AA "LocMgr on %d: "
#  define AB , CkMyPe()
#  define DEBUG(x) CkPrintf x
#  define DEBAD(x) CkPrintf x
#else
#  define DEB(X)   /*CkPrintf x*/
#  define DEBI(X)  /*CkPrintf x*/
#  define DEBC(X)  /*CkPrintf x*/
#  define DEBS(x)  /*CkPrintf x*/
#  define DEBM(X)  /*CkPrintf x*/
#  define DEBL(X)  /*CkPrintf x*/
#  define DEBN(x)  /*CkPrintf x*/
#  define DEBB(x)  /*CkPrintf x*/
#  define str(x)   /**/
#  define DEBUG(x) /**/
#  define DEBAD(x) /*CkPrintf x*/
#endif

// whether to use block mapping in the SMP node level
bool useNodeBlkMapping;

/// Message size above which the runtime will buffer messages directed at
/// unlocated array elements
int _messageBufferingThreshold;

#if CMK_LBDB_ON

#  if CMK_GLOBAL_LOCATION_UPDATE
void UpdateLocation(MigrateInfo& migData)
{
  // CmiPrintf("calls update location\n");
  CkGroupID locMgrGid = ck::ObjID(migData.obj.id).getCollectionID();
  CkLocMgr* localLocMgr = (CkLocMgr*)CkLocalBranch(locMgrGid);
  CkLocCache *cache = (CkLocCache *)CkLocalBranch(localLocMgr->getLocationCache());

  CmiUInt8 elementID = ck::ObjID(migData.obj.id).getElementID();
  // A PE that has never dealt with this element cannot name its index (see
  // CkLocMgr::tryLookupIdx) and has no cache entry worth refreshing, so there
  // is nothing to do here. That is the common case for the bystander PEs a
  // broadcast location update reaches, and asserting on them made migration
  // of any non-compressible (e.g. multidimensional) index fatal.
  CkArrayIndex idx;
  const bool haveIdx = localLocMgr->tryLookupIdx(elementID, idx);

  CkLocEntry entry;
  entry.id = elementID;
  entry.pe = migData.to_pe;
  entry.epoch = cache->getEpoch(elementID) + 1;

  // CkPrintf("[%d] UpdateLocation: obj id=%llu from_pe=%d to_pe=%d epoch=%d\n",
  //          CkMyPe(), entry.id, migData.from_pe, entry.pe, entry.epoch);
  if (haveIdx)
  {
    localLocMgr->updateLocation(idx, entry);
  }
  else
  {
    // The index is unrecoverable here, but the location cache is keyed by id,
    // so the new PE can still be recorded. This is the case that matters most:
    // a PE which cannot name the index is exactly one that has only ever sent
    // to the element, and without this its next send resolves the destination
    // to -1 and has to fall back to the topology-agnostic staging path. That
    // is what made every post-migration step slower than no balancing at all.
    cache->updateLocation(entry);
  }
}
#  endif

#endif

/*********************** Array Messages ************************/
CmiUInt8 CkArrayMessage::array_element_id(void)
{
  return ck::ObjID(UsrToEnv((void*)this)->getRecipientID()).getElementID();
}
unsigned short& CkArrayMessage::array_ep(void)
{
  return UsrToEnv((void*)this)->getsetArrayEp();
}
unsigned short& CkArrayMessage::array_ep_bcast(void)
{
  return UsrToEnv((void*)this)->getsetArrayBcastEp();
}
// A small self-message carrying the identity of an element whose deferred
// migration is ready to start. Identity, not the CkLocRec pointer: the record
// can die between enqueue and handling. Used by both the send-completion
// release path and the AtSyncWait park.
struct DeferredMigrateMsg
{
  char header[CmiMsgHeaderSizeBytes];
  CkGroupID mgrGid;
  CmiUInt8 id;
  int toPe;
};

int _deferredMigrateHandlerIdx;

unsigned char& CkArrayMessage::array_hops(void)
{
  return UsrToEnv((void*)this)->getsetArrayHops();
}
unsigned int CkArrayMessage::array_getSrcPe(void)
{
  return UsrToEnv((void*)this)->getsetArraySrcPe();
}
unsigned int CkArrayMessage::array_ifNotThere(void)
{
  return UsrToEnv((void*)this)->getArrayIfNotThere();
}
void CkArrayMessage::array_setIfNotThere(unsigned int i)
{
  UsrToEnv((void*)this)->setArrayIfNotThere(i);
}

/*********************** Array Map ******************
Given an array element index, an array map tells us
the index's "home" Pe.  This is the Pe the element will
be created on, and also where messages to this element will
be forwarded by default.
*/

CkArrayMap::CkArrayMap(void) {}
CkArrayMap::~CkArrayMap() {}
int CkArrayMap::registerArray(const CkArrayIndex& numElements, CkArrayID aid)
{
  return 0;
}
void CkArrayMap::unregisterArray(int idx) {}

#define CKARRAYMAP_POPULATE_INITIAL(POPULATE_CONDITION)                                  \
  int i;                                                                                 \
  int index[6];                                                                          \
  int start_data[6], end_data[6], step_data[6];                                          \
  for (int d = 0; d < 6; d++)                                                            \
  {                                                                                      \
    start_data[d] = 0;                                                                   \
    end_data[d] = step_data[d] = 1;                                                      \
    if (end.dimension >= 4 && d < end.dimension)                                         \
    {                                                                                    \
      start_data[d] = ((short int*)start.data())[d];                                     \
      end_data[d] = ((short int*)end.data())[d];                                         \
      step_data[d] = ((short int*)step.data())[d];                                       \
    }                                                                                    \
    else if (d < end.dimension)                                                          \
    {                                                                                    \
      start_data[d] = start.data()[d];                                                   \
      end_data[d] = end.data()[d];                                                       \
      step_data[d] = step.data()[d];                                                     \
    }                                                                                    \
  }                                                                                      \
                                                                                         \
  for (index[0] = start_data[0]; index[0] < end_data[0]; index[0] += step_data[0])       \
  {                                                                                      \
    for (index[1] = start_data[1]; index[1] < end_data[1]; index[1] += step_data[1])     \
    {                                                                                    \
      for (index[2] = start_data[2]; index[2] < end_data[2]; index[2] += step_data[2])   \
      {                                                                                  \
        for (index[3] = start_data[3]; index[3] < end_data[3]; index[3] += step_data[3]) \
        {                                                                                \
          for (index[4] = start_data[4]; index[4] < end_data[4];                         \
               index[4] += step_data[4])                                                 \
          {                                                                              \
            for (index[5] = start_data[5]; index[5] < end_data[5];                       \
                 index[5] += step_data[5])                                               \
            {                                                                            \
              if (end.dimension == 1)                                                    \
              {                                                                          \
                i = index[0];                                                            \
                CkArrayIndex1D idx(index[0]);                                            \
                if (POPULATE_CONDITION)                                                  \
                {                                                                        \
                  mgr->insertInitial(idx, CkCopyMsg(&ctorMsg));                          \
                }                                                                        \
              }                                                                          \
              else if (end.dimension == 2)                                               \
              {                                                                          \
                i = index[0] * end_data[1] + index[1];                                   \
                CkArrayIndex2D idx(index[0], index[1]);                                  \
                if (POPULATE_CONDITION)                                                  \
                {                                                                        \
                  mgr->insertInitial(idx, CkCopyMsg(&ctorMsg));                          \
                }                                                                        \
              }                                                                          \
              else if (end.dimension == 3)                                               \
              {                                                                          \
                i = (index[0] * end_data[1] + index[1]) * end_data[2] + index[2];        \
                CkArrayIndex3D idx(index[0], index[1], index[2]);                        \
                if (POPULATE_CONDITION)                                                  \
                {                                                                        \
                  mgr->insertInitial(idx, CkCopyMsg(&ctorMsg));                          \
                }                                                                        \
              }                                                                          \
              else if (end.dimension == 4)                                               \
              {                                                                          \
                i = ((index[0] * end_data[1] + index[1]) * end_data[2] + index[2]) *     \
                        end_data[3] +                                                    \
                    index[3];                                                            \
                CkArrayIndex4D idx(index[0], index[1], index[2], index[3]);              \
                if (POPULATE_CONDITION)                                                  \
                {                                                                        \
                  mgr->insertInitial(idx, CkCopyMsg(&ctorMsg));                          \
                }                                                                        \
              }                                                                          \
              else if (end.dimension == 5)                                               \
              {                                                                          \
                i = (((index[0] * end_data[1] + index[1]) * end_data[2] + index[2]) *    \
                         end_data[3] +                                                   \
                     index[3]) *                                                         \
                        end_data[4] +                                                    \
                    index[4];                                                            \
                CkArrayIndex5D idx(index[0], index[1], index[2], index[3], index[4]);    \
                if (POPULATE_CONDITION)                                                  \
                {                                                                        \
                  mgr->insertInitial(idx, CkCopyMsg(&ctorMsg));                          \
                }                                                                        \
              }                                                                          \
              else if (end.dimension == 6)                                               \
              {                                                                          \
                i = ((((index[0] * end_data[1] + index[1]) * end_data[2] + index[2]) *   \
                          end_data[3] +                                                  \
                      index[3]) *                                                        \
                         end_data[4] +                                                   \
                     index[4]) *                                                         \
                        end_data[5] +                                                    \
                    index[5];                                                            \
                CkArrayIndex6D idx(index[0], index[1], index[2], index[3], index[4],     \
                                   index[5]);                                            \
                if (POPULATE_CONDITION)                                                  \
                {                                                                        \
                  mgr->insertInitial(idx, CkCopyMsg(&ctorMsg));                          \
                }                                                                        \
              }                                                                          \
            }                                                                            \
          }                                                                              \
        }                                                                                \
      }                                                                                  \
    }                                                                                    \
  }

void CkArrayMap::populateInitial(int arrayHdl, CkArrayOptions& options, void* ctorMsg,
                                 CkArray* mgr)
{
  CkArrayIndex start = options.getStart();
  CkArrayIndex end = options.getEnd();
  CkArrayIndex step = options.getStep();
  if (end.nInts == 0)
  {
    CkFreeMsg(ctorMsg);
    return;
  }
  int thisPe = CkMyPe();
  /* The CkArrayIndex is supposed to have at most 3 dimensions, which
     means that all the fields are ints, and numElements.nInts represents
     how many of them are used */
  CKARRAYMAP_POPULATE_INITIAL(CMK_RANK_0(procNum(arrayHdl, idx)) == thisPe);

  CkFreeMsg(ctorMsg);
}

void CkArrayMap::storeCkArrayOpts(CkArrayOptions options)
{
  // options will not be used on demand_creation arrays
  storeOpts = options;
}

void CkArrayMap::pup(PUP::er& p)
{
  p | storeOpts;
  p | dynamicIns;
}

CkGroupID _defaultArrayMapID;
CkGroupID _fastArrayMapID;

class RRMap : public CkArrayMap
{
private:
  CkArrayIndex maxIndex;
  uint64_t products[2 * CK_ARRAYINDEX_MAXLEN];
  bool productsInit;

public:
  RRMap(void)
  {
    DEBC((AA "Creating RRMap\n" AB));
    productsInit = false;
  }
  RRMap(CkMigrateMessage* m) : CkArrayMap(m) {}

  void indexInit()
  {
    productsInit = true;
    maxIndex = storeOpts.getEnd();
    products[maxIndex.dimension - 1] = 1;
    if (maxIndex.dimension <= CK_ARRAYINDEX_MAXLEN)
    {
      for (int dim = maxIndex.dimension - 2; dim >= 0; dim--)
      {
        products[dim] = products[dim + 1] * maxIndex.index[dim + 1];
      }
    }
    else
    {
      for (int dim = maxIndex.dimension - 2; dim >= 0; dim--)
      {
        products[dim] = products[dim + 1] * maxIndex.indexShorts[dim + 1];
      }
    }
  }  // End of indexInit

  int procNum(int arrayHdl, const CkArrayIndex& i)
  {
    if (i.dimension == 1)
    {
      // Map 1D integer indices in simple round-robin fashion
      int ans = (i.data()[0]) % CkNumPes();
      return ans;
    }
    else
    {
      if (dynamicIns.find(arrayHdl) != dynamicIns.end())
      {
        // Finding indicates that current array uses dynamic insertion
        // Map other indices based on their hash code, mod a big prime.
        unsigned int hash = (i.hash() + 739) % 1280107;
        int ans = (hash % CkNumPes());
        return ans;
      }
      else
      {
        if (!productsInit)
        {
          indexInit();
        }

        int indexOffset = 0;
        if (i.dimension <= CK_ARRAYINDEX_MAXLEN)
        {
          for (int dim = i.dimension - 1; dim >= 0; dim--)
          {
            indexOffset += (i.index[dim] * products[dim]);
          }
        }
        else
        {
          for (int dim = maxIndex.dimension - 1; dim >= 0; dim--)
          {
            indexOffset += (i.indexShorts[dim] * products[dim]);
          }
        }
        return indexOffset % CkNumPes();
      }
    }
  }

  void pup(PUP::er& p)
  {
    CkArrayMap::pup(p);
    p | maxIndex;
    p | productsInit;
    PUParray(p, products, 2 * CK_ARRAYINDEX_MAXLEN);
  }
};

/**
 * Class used to store the dimensions of the array and precalculate numChares,
 * binSize and other values for the DefaultArrayMap -- ASB
 */
class arrayMapInfo
{
public:
  CkArrayIndex _nelems;
  int _binSizeFloor; /* floor of numChares/numPes */
  int _binSizeCeil;  /* ceiling of numChares/numPes */
  int _numChares;    /* initial total number of chares */
  int _remChares;    /* numChares % numPes -- equals the number of
                        processors in the first set */
  int _numFirstSet;  /* _remChares X (_binSize + 1) -- number of
                        chares in the first set */

  int _nBinSizeFloor; /* floor of numChares/numNodes */
  int _nRemChares;    /* numChares % numNodes -- equals the number of
                         nodes in the first set */
  int _nNumFirstSet;  /* _remChares X (_binSize + 1) -- number of
                         chares in the first set of nodes */

  /** All processors are divided into two sets. Processors in the first set
   *  have one chare more than the processors in the second set. */

  arrayMapInfo(void) {}

  arrayMapInfo(const CkArrayIndex& n) : _nelems(n), _numChares(0) { compute_binsize(); }

  ~arrayMapInfo() {}

  void compute_binsize()
  {
    int numPes = CkNumPes();
    // Now assuming homogenous nodes where each node has the same number of PEs
    int numNodes = CkNumNodes();

    if (_nelems.dimension == 1)
    {
      _numChares = _nelems.data()[0];
    }
    else if (_nelems.dimension == 2)
    {
      _numChares = _nelems.data()[0] * _nelems.data()[1];
    }
    else if (_nelems.dimension == 3)
    {
      _numChares = _nelems.data()[0] * _nelems.data()[1] * _nelems.data()[2];
    }
    else if (_nelems.dimension == 4)
    {
      _numChares =
          (int)(((short int*)_nelems.data())[0] * ((short int*)_nelems.data())[1] *
                ((short int*)_nelems.data())[2] * ((short int*)_nelems.data())[3]);
    }
    else if (_nelems.dimension == 5)
    {
      _numChares =
          (int)(((short int*)_nelems.data())[0] * ((short int*)_nelems.data())[1] *
                ((short int*)_nelems.data())[2] * ((short int*)_nelems.data())[3] *
                ((short int*)_nelems.data())[4]);
    }
    else if (_nelems.dimension == 6)
    {
      _numChares =
          (int)(((short int*)_nelems.data())[0] * ((short int*)_nelems.data())[1] *
                ((short int*)_nelems.data())[2] * ((short int*)_nelems.data())[3] *
                ((short int*)_nelems.data())[4] * ((short int*)_nelems.data())[5]);
    }

    _remChares = _numChares % numPes;
    _binSizeFloor = (int)floor((double)_numChares / (double)numPes);
    _binSizeCeil = (int)ceil((double)_numChares / (double)numPes);
    _numFirstSet = _remChares * (_binSizeFloor + 1);

    _nRemChares = _numChares % numNodes;
    _nBinSizeFloor = _numChares / numNodes;
    _nNumFirstSet = _nRemChares * (_nBinSizeFloor + 1);
  }

  void pup(PUP::er& p)
  {
    p | _nelems;
    p | _binSizeFloor;
    p | _binSizeCeil;
    p | _numChares;
    p | _remChares;
    p | _numFirstSet;
    p | _nBinSizeFloor;
    p | _nRemChares;
    p | _nNumFirstSet;
  }
};

/**
 * The default map object -- This does blocked mapping in the general case and
 * calls the round-robin procNum for the dynamic insertion case -- ASB
 */
class DefaultArrayMap : public RRMap
{
public:
  /** This array stores information about different chare arrays in a Charm
   *  program (dimensions, binsize, numChares etc ... ) */
  CkPupPtrVec<arrayMapInfo> amaps;

public:
  DefaultArrayMap(void) { DEBC((AA "Creating DefaultArrayMap\n" AB)); }

  DefaultArrayMap(CkMigrateMessage* m) : RRMap(m) {}

  int registerArray(const CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx = amaps.size();
    amaps.resize(idx + 1);
    amaps[idx] = new arrayMapInfo(numElements);
    return idx;
  }

  void unregisterArray(int idx)
  {
    delete amaps[idx];
    amaps[idx] = NULL;
  }

  int procNum(int arrayHdl, const CkArrayIndex& i)
  {
    int flati;
    auto& info = amaps[arrayHdl];
    if (info->_nelems.dimension == 0)
    {
      dynamicIns[arrayHdl] = true;
      return RRMap::procNum(arrayHdl, i);
    }

    if (i.dimension == 1)
    {
      flati = i.data()[0];
    }
    else if (i.dimension == 2)
    {
      flati = i.data()[0] * info->_nelems.data()[1] + i.data()[1];
    }
    else if (i.dimension == 3)
    {
      flati = (i.data()[0] * info->_nelems.data()[1] + i.data()[1]) *
                  info->_nelems.data()[2] +
              i.data()[2];
    }
    else if (i.dimension == 4)
    {
      flati = (int)(((((short int*)i.data())[0] *
                          ((short int*)info->_nelems.data())[1] +
                      ((short int*)i.data())[1]) *
                         ((short int*)info->_nelems.data())[2] +
                     ((short int*)i.data())[2]) *
                        ((short int*)info->_nelems.data())[3] +
                    ((short int*)i.data())[3]);
    }
    else if (i.dimension == 5)
    {
      flati = (int)((((((short int*)i.data())[0] *
                           ((short int*)info->_nelems.data())[1] +
                       ((short int*)i.data())[1]) *
                          ((short int*)info->_nelems.data())[2] +
                      ((short int*)i.data())[2]) *
                         ((short int*)info->_nelems.data())[3] +
                     ((short int*)i.data())[3]) *
                        ((short int*)info->_nelems.data())[4] +
                    ((short int*)i.data())[4]);
    }
    else if (i.dimension == 6)
    {
      flati = (int)(((((((short int*)i.data())[0] *
                            ((short int*)info->_nelems.data())[1] +
                        ((short int*)i.data())[1]) *
                           ((short int*)info->_nelems.data())[2] +
                       ((short int*)i.data())[2]) *
                          ((short int*)info->_nelems.data())[3] +
                      ((short int*)i.data())[3]) *
                         ((short int*)info->_nelems.data())[4] +
                     ((short int*)i.data())[4]) *
                        ((short int*)info->_nelems.data())[5] +
                    ((short int*)i.data())[5]);
    }
#if CMK_ERROR_CHECKING
    else
    {
      CkAbort("CkArrayIndex has more than 6 dimensions!");
    }
#endif

    if (useNodeBlkMapping)
    {
      if (flati < info->_numChares)
      {
        int numCharesOnNode = info->_nBinSizeFloor;
        int startNodeID, offsetInNode;
        if (flati < info->_nNumFirstSet)
        {
          numCharesOnNode++;
          startNodeID = flati / numCharesOnNode;
          offsetInNode = flati % numCharesOnNode;
        }
        else
        {
          startNodeID = info->_nRemChares +
                        (flati - info->_nNumFirstSet) / numCharesOnNode;
          offsetInNode = (flati - info->_nNumFirstSet) % numCharesOnNode;
        }
        int nodeSize = CkMyNodeSize();  // assuming every node has same number of PEs
        int elemsPerPE = numCharesOnNode / nodeSize;
        int remElems = numCharesOnNode % nodeSize;
        int firstSetPEs = remElems * (elemsPerPE + 1);
        if (offsetInNode < firstSetPEs)
        {
          return CkNodeFirst(startNodeID) + offsetInNode / (elemsPerPE + 1);
        }
        else
        {
          return CkNodeFirst(startNodeID) + remElems +
                 (offsetInNode - firstSetPEs) / elemsPerPE;
        }
      }
      else
        return (flati % CkNumPes());
    }
    // regular PE-based block mapping
    if (flati < info->_numFirstSet)
      return (flati / (info->_binSizeFloor + 1));
    else if (flati < info->_numChares)
      return (info->_remChares +
              (flati - info->_numFirstSet) / (info->_binSizeFloor));
    else
      return (flati % CkNumPes());
  }

  void pup(PUP::er& p)
  {
    RRMap::pup(p);
    int npes = CkNumPes();
    p | npes;
    p | amaps;
    if (p.isUnpacking() && npes != CkNumPes())
    {  // binSize needs update
      for (int i = 0; i < amaps.size(); i++)
        if (amaps[i])
          amaps[i]->compute_binsize();
    }
  }
};

/**
 *  A fast map for chare arrays which do static insertions and promise NOT
 *  to do late insertions -- ASB
 */
class FastArrayMap : public DefaultArrayMap
{
public:
  FastArrayMap(void) { DEBC((AA "Creating FastArrayMap\n" AB)); }

  FastArrayMap(CkMigrateMessage* m) : DefaultArrayMap(m) {}

  int registerArray(const CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx;
    idx = DefaultArrayMap::registerArray(numElements, aid);

    return idx;
  }

  int procNum(int arrayHdl, const CkArrayIndex& i)
  {
    int flati = 0;
    if (amaps[arrayHdl]->_nelems.dimension == 0)
    {
      return RRMap::procNum(arrayHdl, i);
    }

    if (i.dimension == 1)
    {
      flati = i.data()[0];
    }
    else if (i.dimension == 2)
    {
      flati = i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1];
    }
    else if (i.dimension == 3)
    {
      flati = (i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1]) *
                  amaps[arrayHdl]->_nelems.data()[2] +
              i.data()[2];
    }
    else if (i.dimension == 4)
    {
      flati = (int)(((((short int*)i.data())[0] *
                          ((short int*)amaps[arrayHdl]->_nelems.data())[1] +
                      ((short int*)i.data())[1]) *
                         ((short int*)amaps[arrayHdl]->_nelems.data())[2] +
                     ((short int*)i.data())[2]) *
                        ((short int*)amaps[arrayHdl]->_nelems.data())[3] +
                    ((short int*)i.data())[3]);
    }
    else if (i.dimension == 5)
    {
      flati = (int)((((((short int*)i.data())[0] *
                           ((short int*)amaps[arrayHdl]->_nelems.data())[1] +
                       ((short int*)i.data())[1]) *
                          ((short int*)amaps[arrayHdl]->_nelems.data())[2] +
                      ((short int*)i.data())[2]) *
                         ((short int*)amaps[arrayHdl]->_nelems.data())[3] +
                     ((short int*)i.data())[3]) *
                        ((short int*)amaps[arrayHdl]->_nelems.data())[4] +
                    ((short int*)i.data())[4]);
    }
    else if (i.dimension == 6)
    {
      flati = (int)(((((((short int*)i.data())[0] *
                            ((short int*)amaps[arrayHdl]->_nelems.data())[1] +
                        ((short int*)i.data())[1]) *
                           ((short int*)amaps[arrayHdl]->_nelems.data())[2] +
                       ((short int*)i.data())[2]) *
                          ((short int*)amaps[arrayHdl]->_nelems.data())[3] +
                      ((short int*)i.data())[3]) *
                         ((short int*)amaps[arrayHdl]->_nelems.data())[4] +
                     ((short int*)i.data())[4]) *
                        ((short int*)amaps[arrayHdl]->_nelems.data())[5] +
                    ((short int*)i.data())[5]);
    }
#if CMK_ERROR_CHECKING
    else
    {
      CkAbort("CkArrayIndex has more than 6 dimensions!");
    }
#endif

    /** binSize used in DefaultArrayMap is the floor of numChares/numPes
     *  but for this FastArrayMap, we need the ceiling */
    return (flati / amaps[arrayHdl]->_binSizeCeil);
  }

  void pup(PUP::er& p) { DefaultArrayMap::pup(p); }
};

/* *
 * Hilbert map object -- This does hilbert mapping.
 * Convert array indices into 1D fashion according to their Hilbert filling curve
 */

typedef struct
{
  int intIndex;
  std::vector<int> coords;
} hilbert_pair;

bool operator==(hilbert_pair p1, hilbert_pair p2) { return p1.intIndex == p2.intIndex; }

bool myCompare(hilbert_pair p1, hilbert_pair p2) { return p1.intIndex < p2.intIndex; }

class HilbertArrayMap : public DefaultArrayMap
{
  std::vector<int> allpairs;
  std::vector<int> procList;

public:
  HilbertArrayMap(void)
  {
    procList.resize(CkNumPes());
    getHilbertList(procList.data());
    DEBC((AA "Creating HilbertArrayMap\n" AB));
  }

  HilbertArrayMap(CkMigrateMessage* m) : DefaultArrayMap(m) {}

  ~HilbertArrayMap() {}

  int registerArray(const CkArrayIndex& i, CkArrayID aid)
  {
    int idx;
    idx = DefaultArrayMap::registerArray(i, aid);

    if (i.dimension == 1)
    {
      // CkPrintf("1D %d\n", amaps[idx]->_nelems.data()[0]);
    }
    else if (i.dimension == 2)
    {
      // CkPrintf("2D %d:%d\n", amaps[idx]->_nelems.data()[0],
      // amaps[idx]->_nelems.data()[1]);
      const int dims = 2;
      int nDim0 = amaps[idx]->_nelems.data()[0];
      int nDim1 = amaps[idx]->_nelems.data()[1];
      int index;
      int counter = 0;
      std::vector<int> coords;
      allpairs.resize((size_t)nDim0 * nDim1);
      coords.resize(dims);
      for (int i = 0; i < nDim0; i++)
        for (int j = 0; j < nDim1; j++)
        {
          coords[0] = i;
          coords[1] = j;
          index = Hilbert_to_int(coords, dims);
          // CkPrintf("(%d:%d)----------> %d \n", i, j, index);
          allpairs[counter] = index;
          counter++;
        }
    }
    else if (i.dimension == 3)
    {
      // CkPrintf("3D %d:%d:%d\n", amaps[idx]->_nelems.data()[0],
      // amaps[idx]->_nelems.data()[1],
      //        amaps[idx]->_nelems.data()[2]);
      const int dims = 3;
      int nDim0 = amaps[idx]->_nelems.data()[0];
      int nDim1 = amaps[idx]->_nelems.data()[1];
      int nDim2 = amaps[idx]->_nelems.data()[2];
      int index;
      int counter = 0;
      std::vector<int> coords;
      allpairs.resize((size_t)nDim0 * nDim1 * nDim2);
      coords.resize(dims);
      for (int i = 0; i < nDim0; i++)
        for (int j = 0; j < nDim1; j++)
          for (int k = 0; k < nDim2; k++)
          {
            coords[0] = i;
            coords[1] = j;
            coords[2] = k;
            index = Hilbert_to_int(coords, dims);
            allpairs[counter] = index;
            counter++;
          }
    }
    else if (i.dimension == 4)
    {
      // CkPrintf("4D %hd:%hd:%hd:%hd\n", ((short int*)amaps[idx]->_nelems.data())[0],
      //        ((short int*)amaps[idx]->_nelems.data())[1], ((short
      //        int*)amaps[idx]->_nelems.data())[2],
      //        ((short int*)amaps[idx]->_nelems.data())[3]);
      const int dims = 4;
      int nDim[dims];
      for (int k = 0; k < dims; k++)
      {
        nDim[k] = (int)((short int*)amaps[idx]->_nelems.data())[k];
      }
      int index;
      int counter = 0;
      std::vector<int> coords;
      allpairs.resize((size_t)nDim[0] * nDim[1] * nDim[2] * nDim[3]);
      coords.resize(dims);
      for (int i = 0; i < nDim[0]; i++)
        for (int j = 0; j < nDim[1]; j++)
          for (int k = 0; k < nDim[2]; k++)
            for (int x = 0; x < nDim[3]; x++)
            {
              coords[0] = i;
              coords[1] = j;
              coords[2] = k;
              coords[3] = x;
              index = Hilbert_to_int(coords, dims);
              allpairs[counter] = index;
              counter++;
            }
    }
    else if (i.dimension == 5)
    {
      // CkPrintf("5D %hd:%hd:%hd:%hd:%hd\n", ((short int*)amaps[idx]->_nelems.data())[0],
      //        ((short int*)amaps[idx]->_nelems.data())[1], ((short
      //        int*)amaps[idx]->_nelems.data())[2],
      //        ((short int*)amaps[idx]->_nelems.data())[3], ((short
      //        int*)amaps[idx]->_nelems.data())[4]);
      const int dims = 5;
      int nDim[dims];
      for (int k = 0; k < dims; k++)
      {
        nDim[k] = (int)((short int*)amaps[idx]->_nelems.data())[k];
      }
      int index;
      int counter = 0;
      std::vector<int> coords;
      allpairs.resize((size_t)nDim[0] * nDim[1] * nDim[2] * nDim[3] * nDim[4]);
      coords.resize(dims);
      for (int i = 0; i < nDim[0]; i++)
        for (int j = 0; j < nDim[1]; j++)
          for (int k = 0; k < nDim[2]; k++)
            for (int x = 0; x < nDim[3]; x++)
              for (int y = 0; y < nDim[4]; y++)
              {
                coords[0] = i;
                coords[1] = j;
                coords[2] = k;
                coords[3] = x;
                coords[4] = y;
                index = Hilbert_to_int(coords, dims);
                allpairs[counter] = index;
                counter++;
              }
    }
    else if (i.dimension == 6)
    {
      // CkPrintf("6D %hd:%hd:%hd:%hd:%hd:%hd\n", ((short
      // int*)amaps[idx]->_nelems.data())[0],
      //        ((short int*)amaps[idx]->_nelems.data())[1], ((short
      //        int*)amaps[idx]->_nelems.data())[2],
      //        ((short int*)amaps[idx]->_nelems.data())[3], ((short
      //        int*)amaps[idx]->_nelems.data())[4],
      //        ((short int*)amaps[idx]->_nelems.data())[5]);
      const int dims = 6;
      int nDim[dims];
      for (int k = 0; k < dims; k++)
      {
        nDim[k] = (int)((short int*)amaps[idx]->_nelems.data())[k];
      }
      int index;
      int counter = 0;
      std::vector<int> coords;
      allpairs.resize((size_t)nDim[0] * nDim[1] * nDim[2] * nDim[3] * nDim[4] * nDim[5]);
      coords.resize(dims);
      for (int i = 0; i < nDim[0]; i++)
        for (int j = 0; j < nDim[1]; j++)
          for (int k = 0; k < nDim[2]; k++)
            for (int x = 0; x < nDim[3]; x++)
              for (int y = 0; y < nDim[4]; y++)
                for (int z = 0; z < nDim[5]; z++)
                {
                  coords[0] = i;
                  coords[1] = j;
                  coords[2] = k;
                  coords[3] = x;
                  coords[4] = y;
                  coords[5] = y;
                  index = Hilbert_to_int(coords, dims);
                  allpairs[counter] = index;
                  counter++;
                }
    }
    return idx;
  }

  int procNum(int arrayHdl, const CkArrayIndex& i)
  {
    int flati = 0;
    int myInt;
    if (amaps[arrayHdl]->_nelems.dimension == 0)
    {
      return RRMap::procNum(arrayHdl, i);
    }
    if (i.dimension == 1)
    {
      flati = i.data()[0];
    }
    else if (i.dimension == 2)
    {
      int nDim1 = amaps[arrayHdl]->_nelems.data()[1];
      myInt = i.data()[0] * nDim1 + i.data()[1];
      flati = allpairs[myInt];
    }
    else if (i.dimension == 3)
    {
      hilbert_pair mypair;
      mypair.coords.resize(3);
      int nDim[2];
      for (int j = 0; j < 2; j++)
      {
        nDim[j] = amaps[arrayHdl]->_nelems.data()[j + 1];
      }
      myInt = i.data()[0] * nDim[0] * nDim[1] + i.data()[1] * nDim[1] + i.data()[2];
      flati = allpairs[myInt];
    }
    else if (i.dimension == 4)
    {
      hilbert_pair mypair;
      mypair.coords.resize(4);
      short int nDim[3];
      for (int j = 0; j < 3; j++)
      {
        nDim[j] = ((short int*)amaps[arrayHdl]->_nelems.data())[j + 1];
      }
      myInt = (int)(((short int*)i.data())[0] * nDim[0] * nDim[1] * nDim[2] +
                    ((short int*)i.data())[1] * nDim[1] * nDim[2] +
                    ((short int*)i.data())[2] * nDim[2] + ((short int*)i.data())[3]);
      flati = allpairs[myInt];
    }
    else if (i.dimension == 5)
    {
      hilbert_pair mypair;
      mypair.coords.resize(5);
      short int nDim[4];
      for (int j = 0; j < 4; j++)
      {
        nDim[j] = ((short int*)amaps[arrayHdl]->_nelems.data())[j + 1];
      }
      myInt = (int)(((short int*)i.data())[0] * nDim[0] * nDim[1] * nDim[2] * nDim[3] +
                    ((short int*)i.data())[1] * nDim[1] * nDim[2] * nDim[3] +
                    ((short int*)i.data())[2] * nDim[2] * nDim[3] +
                    ((short int*)i.data())[3] * nDim[3] + ((short int*)i.data())[4]);
      flati = allpairs[myInt];
    }
    else if (i.dimension == 6)
    {
      hilbert_pair mypair;
      mypair.coords.resize(6);
      short int nDim[5];
      for (int j = 0; j < 5; j++)
      {
        nDim[j] = ((short int*)amaps[arrayHdl]->_nelems.data())[j + 1];
      }
      myInt = (int)(((short int*)i.data())[0] * nDim[0] * nDim[1] * nDim[2] * nDim[3] *
                        nDim[4] +
                    ((short int*)i.data())[1] * nDim[1] * nDim[2] * nDim[3] * nDim[4] +
                    ((short int*)i.data())[2] * nDim[2] * nDim[3] * nDim[4] +
                    ((short int*)i.data())[3] * nDim[3] * nDim[4] +
                    ((short int*)i.data())[4] * nDim[4] + ((short int*)i.data())[5]);
      flati = allpairs[myInt];
    }
#if CMK_ERROR_CHECKING
    else
    {
      CkAbort("CkArrayIndex has more than 6 dimensions!");
    }
#endif

    /** binSize used in DefaultArrayMap is the floor of numChares/numPes
     *  but for this FastArrayMap, we need the ceiling */
    int block = flati / amaps[arrayHdl]->_binSizeCeil;
    // for(int i=0; i<CkNumPes(); i++)
    //    CkPrintf("(%d:%d) ", i, procList[i]);
    // CkPrintf("\n");
    // CkPrintf("block [%d:%d]\n", block, procList[block]);
    return procList[block];
  }

  void pup(PUP::er& p) { DefaultArrayMap::pup(p); }
};

/**
 * This map can be used for topology aware mapping when the mapping is provided
 * through a file -- ASB
 */
class ReadFileMap : public DefaultArrayMap
{
private:
  std::vector<int> mapping;

public:
  ReadFileMap(void) { DEBC((AA "Creating ReadFileMap\n" AB)); }

  ReadFileMap(CkMigrateMessage* m) : DefaultArrayMap(m) {}

  int registerArray(const CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx;
    idx = DefaultArrayMap::registerArray(numElements, aid);

    if (mapping.size() == 0)
    {
      int numChares;

      if (amaps[idx]->_nelems.dimension == 1)
      {
        numChares = amaps[idx]->_nelems.data()[0];
      }
      else if (amaps[idx]->_nelems.dimension == 2)
      {
        numChares = amaps[idx]->_nelems.data()[0] * amaps[idx]->_nelems.data()[1];
      }
      else if (amaps[idx]->_nelems.dimension == 3)
      {
        numChares = amaps[idx]->_nelems.data()[0] * amaps[idx]->_nelems.data()[1] *
                    amaps[idx]->_nelems.data()[2];
      }
      else if (amaps[idx]->_nelems.dimension == 4)
      {
        numChares = (int)(((short int*)amaps[idx]->_nelems.data())[0] *
                          ((short int*)amaps[idx]->_nelems.data())[1] *
                          ((short int*)amaps[idx]->_nelems.data())[2] *
                          ((short int*)amaps[idx]->_nelems.data())[3]);
      }
      else if (amaps[idx]->_nelems.dimension == 5)
      {
        numChares = (int)(((short int*)amaps[idx]->_nelems.data())[0] *
                          ((short int*)amaps[idx]->_nelems.data())[1] *
                          ((short int*)amaps[idx]->_nelems.data())[2] *
                          ((short int*)amaps[idx]->_nelems.data())[3] *
                          ((short int*)amaps[idx]->_nelems.data())[4]);
      }
      else if (amaps[idx]->_nelems.dimension == 6)
      {
        numChares = (int)(((short int*)amaps[idx]->_nelems.data())[0] *
                          ((short int*)amaps[idx]->_nelems.data())[1] *
                          ((short int*)amaps[idx]->_nelems.data())[2] *
                          ((short int*)amaps[idx]->_nelems.data())[3] *
                          ((short int*)amaps[idx]->_nelems.data())[4] *
                          ((short int*)amaps[idx]->_nelems.data())[5]);
      }
      else
      {
        CkAbort("CkArrayIndex has more than 6 dimension!");
      }

      mapping.resize(numChares);
      FILE* mapf = fopen("mapfile", "r");
      TopoManager tmgr;
      int x, y, z, t;

      for (int i = 0; i < numChares; i++)
      {
        if (fscanf(mapf, "%d %d %d %d", &x, &y, &z, &t) != 4)
        {
          CkAbort("ReadFileMap> reading from mapfile failed!");
        }
        mapping[i] = tmgr.coordinatesToRank(x, y, z, t);
      }
      fclose(mapf);
    }

    return idx;
  }

  int procNum(int arrayHdl, const CkArrayIndex& i)
  {
    int flati;

    if (i.dimension == 1)
    {
      flati = i.data()[0];
    }
    else if (i.dimension == 2)
    {
      flati = i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1];
    }
    else if (i.dimension == 3)
    {
      flati = (i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1]) *
                  amaps[arrayHdl]->_nelems.data()[2] +
              i.data()[2];
    }
    else if (i.dimension == 4)
    {
      flati = (int)(((((short int*)i.data())[0] *
                          ((short int*)amaps[arrayHdl]->_nelems.data())[1] +
                      ((short int*)i.data())[1]) *
                         ((short int*)amaps[arrayHdl]->_nelems.data())[2] +
                     ((short int*)i.data())[2]) *
                        ((short int*)amaps[arrayHdl]->_nelems.data())[3] +
                    ((short int*)i.data())[3]);
    }
    else if (i.dimension == 5)
    {
      flati = (int)((((((short int*)i.data())[0] *
                           ((short int*)amaps[arrayHdl]->_nelems.data())[1] +
                       ((short int*)i.data())[1]) *
                          ((short int*)amaps[arrayHdl]->_nelems.data())[2] +
                      ((short int*)i.data())[2]) *
                         ((short int*)amaps[arrayHdl]->_nelems.data())[3] +
                     ((short int*)i.data())[3]) *
                        ((short int*)amaps[arrayHdl]->_nelems.data())[4] +
                    ((short int*)i.data())[4]);
    }
    else if (i.dimension == 6)
    {
      flati = (int)(((((((short int*)i.data())[0] *
                            ((short int*)amaps[arrayHdl]->_nelems.data())[1] +
                        ((short int*)i.data())[1]) *
                           ((short int*)amaps[arrayHdl]->_nelems.data())[2] +
                       ((short int*)i.data())[2]) *
                          ((short int*)amaps[arrayHdl]->_nelems.data())[3] +
                      ((short int*)i.data())[3]) *
                         ((short int*)amaps[arrayHdl]->_nelems.data())[4] +
                     ((short int*)i.data())[4]) *
                        ((short int*)amaps[arrayHdl]->_nelems.data())[5] +
                    ((short int*)i.data())[5]);
    }
    else
    {
      CkAbort("CkArrayIndex has more than 6 dimensions!");
    }

    return mapping[flati];
  }

  void pup(PUP::er& p)
  {
    DefaultArrayMap::pup(p);
    p | mapping;
  }
};

/**
 * This map can be used for a simple (non-topology-aware) mapping of a 1D array when the
 * mapping is provided through a file
 */
class Simple1DFileMap : public DefaultArrayMap
{
private:
  std::vector<int> mapping;

public:
  Simple1DFileMap(void) {
    DEBC((AA "Creating Simple1DFileMap\n" AB));
  }

  Simple1DFileMap(CkMigrateMessage *m) : DefaultArrayMap(m){}

  int registerArray(const CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx = DefaultArrayMap::registerArray(numElements, aid);

    if(mapping.size() == 0) {
      int numChares;

      if (amaps[idx]->_nelems.dimension == 1) {
        numChares = amaps[idx]->_nelems.data()[0];
      } else {
        CkAbort("CkArrayIndex has more than 1 dimension for a Simple1DFileMap!");
      }

      mapping.resize(numChares);
      FILE *mapf = fopen("mapfile", "r");
      if (mapf == NULL) {
        CkAbort("Simple1DFileMap failed to open file named 'mapfile'!");
      }
      int pe;

      for(int i=0; i<numChares; i++) {
        if (fscanf(mapf, "%d\n", &pe) != 1) {
          CkAbort("Simple1DFileMap> reading from mapfile failed! Expected one int per line, one line per chare array element...");
        }
        mapping[i] = pe;
      }
      fclose(mapf);
    }

    return idx;
  }

  int procNum(int arrayHdl, const CkArrayIndex &i) {
    int flati;

    if (i.dimension == 1) {
      flati = i.data()[0];
    } else {
      CkAbort("CkArrayIndex has more than 1 dimension for a 1D map!");
    }

    return mapping[flati];
  }

  void pup(PUP::er& p){
    DefaultArrayMap::pup(p);
    p|mapping;
  }
};

// This is currently  here for backwards compatibility
class BlockMap : public DefaultArrayMap
{
public:
  BlockMap() {}
  BlockMap(CkMigrateMessage* m) : DefaultArrayMap(m) {}
};

/**
 * map object-- use seed load balancer.
 */
class CldMap : public CkArrayMap
{
public:
  CldMap(void) { DEBC((AA "Creating CldMap\n" AB)); }
  CldMap(CkMigrateMessage* m) : CkArrayMap(m) {}
  int homePe(int /*arrayHdl*/, const CkArrayIndex& i)
  {
    if (i.dimension == 1)
    {
      // Map 1D integer indices in simple round-robin fashion
      return (i.data()[0]) % CkNumPes();
    }
    else
    {
      // Map other indices based on their hash code, mod a big prime.
      unsigned int hash = (i.hash() + 739) % 1280107;
      return (hash % CkNumPes());
    }
  }
  int procNum(int arrayHdl, const CkArrayIndex& i)
  {
    return CLD_ANYWHERE;  // -1
  }
  void populateInitial(int arrayHdl, CkArrayOptions& options, void* ctorMsg, CkArray* mgr)
  {
    CkArrayIndex start = options.getStart();
    CkArrayIndex end = options.getEnd();
    CkArrayIndex step = options.getStep();
    if (end.dimension == 0)
    {
      CkFreeMsg(ctorMsg);
      return;
    }
    int thisPe = CkMyPe();
    int numPes = CkNumPes();

    CKARRAYMAP_POPULATE_INITIAL(i % numPes == thisPe);
    CkFreeMsg(ctorMsg);
  }
};

/// A class responsible for parsing the command line arguments for the PE
/// to extract the format string passed in with +ConfigurableRRMap
class ConfigurableRRMapLoader
{
public:
  std::vector<int> locations;
  int objs_per_block;
  int PE_per_block;

  /// labels for states used when parsing the ConfigurableRRMap from ARGV
  enum ConfigurableRRMapLoadStatus : uint8_t
  {
    not_loaded,
    loaded_found,
    loaded_not_found
  };

  enum ConfigurableRRMapLoadStatus state;

  ConfigurableRRMapLoader()
  {
    state = not_loaded;
    objs_per_block = 0;
    PE_per_block = 0;
  }

  /// load configuration if possible, and return whether a valid configuration exists
  bool haveConfiguration()
  {
    if (state == not_loaded)
    {
      DEBUG(("[%d] loading ConfigurableRRMap configuration\n", CkMyPe()));
      char** argv = CkGetArgv();
      char* configuration = NULL;
      bool found = CmiGetArgString(argv, "+ConfigurableRRMap", &configuration);
      if (!found)
      {
        DEBUG(("Couldn't find +ConfigurableRRMap command line argument\n"));
        state = loaded_not_found;
        return false;
      }
      else
      {
        DEBUG(("Found +ConfigurableRRMap command line argument in %p=\"%s\"\n",
               configuration, configuration));

        std::istringstream instream(configuration);
        CkAssert(instream.good());

        // Example line:
        // 10 8 0 1 2 3 4 5 6 7 7 7 7
        // Map 10 objects to 8 PEs, with each object's index among the 8 PEs.

        // extract first integer
        instream >> objs_per_block >> PE_per_block;
        CkAssert(instream.good());
        CkAssert(objs_per_block > 0);
        CkAssert(PE_per_block > 0);
        locations.resize(objs_per_block);
        for (int i = 0; i < objs_per_block; i++)
        {
          locations[i] = 0;
          CkAssert(instream.good());
          instream >> locations[i];
          CkAssert(locations[i] < PE_per_block);
        }
        state = loaded_found;
        return true;
      }
    }
    else
    {
      DEBUG(("[%d] ConfigurableRRMap has already been loaded\n", CkMyPe()));
      return state == loaded_found;
    }
  }
};

CkpvDeclare(ConfigurableRRMapLoader, myConfigRRMapState);

void _initConfigurableRRMap()
{
  CkpvInitialize(ConfigurableRRMapLoader, myConfigRRMapState);
}

/// Try to load the command line arguments for ConfigurableRRMap
bool haveConfigurableRRMap()
{
  DEBUG(("haveConfigurableRRMap()\n"));
  ConfigurableRRMapLoader& loader = CkpvAccess(myConfigRRMapState);
  return loader.haveConfiguration();
}

class ConfigurableRRMap : public RRMap
{
public:
  ConfigurableRRMap(void) { DEBC((AA "Creating ConfigurableRRMap\n" AB)); }
  ConfigurableRRMap(CkMigrateMessage* m) : RRMap(m) {}

  void populateInitial(int arrayHdl, CkArrayOptions& options, void* ctorMsg, CkArray* mgr)
  {
    CkArrayIndex end = options.getEnd();
    // Try to load the configuration from command line argument
    CkAssert(haveConfigurableRRMap());
    ConfigurableRRMapLoader& loader = CkpvAccess(myConfigRRMapState);
    if (end.dimension == 0)
    {
      CkFreeMsg(ctorMsg);
      return;
    }
    int thisPe = CkMyPe();
    int maxIndex = end.data()[0];
    DEBUG(("[%d] ConfigurableRRMap: index=%d,%d,%d\n", CkMyPe(), (int)end.data()[0],
           (int)end.data()[1], (int)end.data()[2]));

    if (end.dimension != 1)
    {
      CkAbort("ConfigurableRRMap only supports dimension 1!");
    }

    for (int index = 0; index < maxIndex; index++)
    {
      CkArrayIndex1D idx(index);

      int cyclic_block = index / loader.objs_per_block;
      int cyclic_local = index % loader.objs_per_block;
      int l = loader.locations[cyclic_local];
      int PE = (cyclic_block * loader.PE_per_block + l) % CkNumPes();

      DEBUG(("[%d] ConfigurableRRMap: index=%d is located on PE %d l=%d\n", CkMyPe(),
             (int)index, (int)PE, l));

      if (PE == thisPe)
        mgr->insertInitial(idx, CkCopyMsg(&ctorMsg));
    }
    //        CKARRAYMAP_POPULATE_INITIAL(PE == thisPe);

    CkFreeMsg(ctorMsg);
  }
};

CkpvStaticDeclare(double*, rem);

class arrInfo
{
private:
  CkArrayIndex _nelems;
  std::vector<int> _map;

public:
  arrInfo() {}
  arrInfo(const CkArrayIndex& n, int* _speeds)
      : _nelems(n), _map(_nelems.getCombinedCount())
  {
    distrib(_speeds);
  }
  ~arrInfo() {}
  int getMap(const CkArrayIndex& i);
  void distrib(int* speeds);
  void pup(PUP::er& p)
  {
    p | _nelems;
    p | _map;
  }
};

static int cmp(const void* first, const void* second)
{
  int fi = *((const int*)first);
  int si = *((const int*)second);
  return ((CkpvAccess(rem)[fi] == CkpvAccess(rem)[si])
              ? 0
              : ((CkpvAccess(rem)[fi] < CkpvAccess(rem)[si]) ? 1 : (-1)));
}

void arrInfo::distrib(int* speeds)
{
  int _nelemsCount = _nelems.getCombinedCount();
  double total = 0.0;
  int npes = CkNumPes();
  int i, j, k;
  for (i = 0; i < npes; i++) total += (double)speeds[i];
  std::vector<double> nspeeds(npes);
  for (i = 0; i < npes; i++) nspeeds[i] = (double)speeds[i] / total;
  std::vector<int> cp(npes);
  for (i = 0; i < npes; i++) cp[i] = (int)(nspeeds[i] * _nelemsCount);
  int nr = 0;
  for (i = 0; i < npes; i++) nr += cp[i];
  nr = _nelemsCount - nr;
  if (nr != 0)
  {
    CkpvAccess(rem) = new double[npes];
    for (i = 0; i < npes; i++)
      CkpvAccess(rem)[i] = (double)_nelemsCount * nspeeds[i] - cp[i];
    std::vector<int> pes(npes);
    for (i = 0; i < npes; i++) pes[i] = i;
    qsort(pes.data(), npes, sizeof(int), cmp);
    for (i = 0; i < nr; i++) cp[pes[i]]++;
    delete[] CkpvAccess(rem);
  }
  k = 0;
  for (i = 0; i < npes; i++)
  {
    for (j = 0; j < cp[i]; j++) _map[k++] = i;
  }
}

int arrInfo::getMap(const CkArrayIndex& i)
{
  if (i.dimension == 1)
    return _map[i.data()[0]];
  else
    return _map[((i.hash() + 739) % 1280107) % _nelems.getCombinedCount()];
}

// Speeds maps processor number to "speed" (some sort of iterations per second counter)
// It is initialized by processor 0.
static int* globalSpeeds;

#if CMK_USE_PROP_MAP
typedef struct _speedmsg
{
  char hdr[CmiMsgHeaderSizeBytes];
  int node;
  int speed;
} speedMsg;

static void _speedHdlr(void* m)
{
  speedMsg* msg = (speedMsg*)m;
  if (CmiMyRank() == 0)
    for (int pe = 0; pe < CmiNodeSize(msg->node); pe++)
      globalSpeeds[CmiNodeFirst(msg->node) + pe] = msg->speed;
  CmiFree(m);
}

// initnode call
void _propMapInit(void)
{
  globalSpeeds = new int[CkNumPes()];
  int hdlr = CkRegisterHandler(_speedHdlr);
  CmiPrintf("[%d]Measuring processor speed for prop. mapping...\n", CkMyPe());
  int s = LBManager::ProcessorSpeed();
  speedMsg msg;
  CmiSetHandler(&msg, hdlr);
  msg.node = CkMyNode();
  msg.speed = s;
  CmiSyncBroadcastAllAndFree(sizeof(msg), &msg);
  for (int i = 0; i < CkNumNodes(); i++) CmiDeliverSpecificMsg(hdlr);
}
#else
void _propMapInit(void)
{
  globalSpeeds = new int[CkNumPes()];
  int i;
  for (i = 0; i < CkNumPes(); i++) globalSpeeds[i] = 1;
}
#endif
/**
 * A proportional map object-- tries to map more objects to
 * faster processors and fewer to slower processors.  Also
 * attempts to ensure good locality by mapping nearby elements
 * together.
 */
class PropMap : public CkArrayMap
{
private:
  CkPupPtrVec<arrInfo> arrs;

public:
  PropMap(void)
  {
    CkpvInitialize(double*, rem);
    DEBC((AA "Creating PropMap\n" AB));
  }
  PropMap(CkMigrateMessage* m) {}
  int registerArray(const CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx = arrs.size();
    arrs.resize(idx + 1);
    arrs[idx] = new arrInfo(numElements, globalSpeeds);
    return idx;
  }
  void unregisterArray(int idx) { arrs[idx].destroy(); }
  int procNum(int arrayHdl, const CkArrayIndex& i) { return arrs[arrayHdl]->getMap(i); }
  void pup(PUP::er& p)
  {
    int oldNumPes = -1;
    if (p.isPacking())
    {
      oldNumPes = CkNumPes();
    }
    p | oldNumPes;
    p | arrs;
    if (p.isUnpacking() && oldNumPes != CkNumPes())
    {
      for (int idx = 0; idx < arrs.length(); ++idx)
      {
        arrs[idx]->distrib(globalSpeeds);
      }
    }
  }
};

class CkMapsInit : public Chare
{
public:
  CkMapsInit(CkArgMsg* msg)
  {
    //_defaultArrayMapID = CProxy_HilbertArrayMap::ckNew();
    _defaultArrayMapID = CProxy_DefaultArrayMap::ckNew();
    _fastArrayMapID = CProxy_FastArrayMap::ckNew();
    delete msg;
  }

  CkMapsInit(CkMigrateMessage* m) {}
};

// given an envelope of a Charm msg, find the recipient object pointer
CkMigratable* CkArrayMessageObjectPtr(envelope* env)
{
  if (env->getMsgtype() != ForArrayEltMsg)
    return NULL;  // not an array msg

  ///@todo: Delegate this to the array manager which can then deal with ForArrayEltMsg
  CkArray* mgr = CProxy_CkArray(env->getArrayMgr()).ckLocalBranch();
  return mgr ? mgr->lookup(ck::ObjID(env->getRecipientID()).getElementID()) : NULL;
}

/****************************** Out-of-Core support ********************/

#if CMK_OUT_OF_CORE
CooPrefetchManager CkArrayElementPrefetcher;
CkpvDeclare(int, CkSaveRestorePrefetch);

/**
 * Return the out-of-core objid (from CooRegisterObject)
 * that this Converse message will access.  If the message
 * will not access an object, return -1.
 */
int CkArrayPrefetch_msg2ObjId(void* msg)
{
  envelope* env = (envelope*)msg;
  CkMigratable* elt = CkArrayMessageObjectPtr(env);
  return elt ? elt->prefetchObjID : -1;
}

/**
 * Write this object (registered with RegisterObject)
 * to this writable file.
 */
void CkArrayPrefetch_writeToSwap(FILE* swapfile, void* objptr)
{
  CkMigratable* elt = (CkMigratable*)objptr;

  // Save the element's data to disk:
  PUP::toDisk p(swapfile);
  elt->virtual_pup(p);

  // Call the element's destructor in-place (so pointer doesn't change)
  CkpvAccess(CkSaveRestorePrefetch) = 1;
  elt->~CkMigratable();  //< because destructor is virtual, destroys user class too.
  CkpvAccess(CkSaveRestorePrefetch) = 0;
}

/**
 * Read this object (registered with RegisterObject)
 * from this readable file.
 */
void CkArrayPrefetch_readFromSwap(FILE* swapfile, void* objptr)
{
  CkMigratable* elt = (CkMigratable*)objptr;
  // Call the element's migration constructor in-place
  CkpvAccess(CkSaveRestorePrefetch) = 1;
  int ctorIdx = _chareTable[elt->thisChareType]->migCtor;
  elt->myRec->invokeEntry(elt, (CkMigrateMessage*)0, ctorIdx, true);
  CkpvAccess(CkSaveRestorePrefetch) = 0;

  // Restore the element's data from disk:
  PUP::fromDisk p(swapfile);
  elt->virtual_pup(p);
}

static void _CkMigratable_prefetchInit(void)
{
  CkpvExtern(int, CkSaveRestorePrefetch);
  CkpvAccess(CkSaveRestorePrefetch) = 0;
  CkArrayElementPrefetcher.msg2ObjId = CkArrayPrefetch_msg2ObjId;
  CkArrayElementPrefetcher.writeToSwap = CkArrayPrefetch_writeToSwap;
  CkArrayElementPrefetcher.readFromSwap = CkArrayPrefetch_readFromSwap;
  CooRegisterManager(&CkArrayElementPrefetcher, _charmHandlerIdx);
}
#endif

/****************************** CkMigratable ***************************/
/**
 * This tiny class is used to convey information to the
 * newly created CkMigratable object when its constructor is called.
 */

CkpvDeclare(CkMigratable_initInfo, mig_initInfo);

void _CkMigratable_initInfoInit(void)
{
  CkpvInitialize(CkMigratable_initInfo, mig_initInfo);
#if CMK_OUT_OF_CORE
  _CkMigratable_prefetchInit();
#endif
}

void CkMigratable::commonInit(void)
{
  CkMigratable_initInfo& i = CkpvAccess(mig_initInfo);
#if CMK_OUT_OF_CORE
  isInCore = true;
  if (CkpvAccess(CkSaveRestorePrefetch))
    return;            /* Just restoring from disk--don't touch object */
  prefetchObjID = -1;  // Unregistered
#endif
  myRec = i.locRec;
  thisIndexMax = myRec->getIndex();
  thisChareType = i.chareType;
  usesAtSync = false;
  usesAutoMeasure = true;
  barrierRegistered = false;

  local_state = OFF;
  prev_load = 0.0;
  can_reset = false;
  lbStepSeen = 0;
  lbIterNo = 0;
  lbStepBlocked = false;
  metaLBStreamJoinedPe = -1;
  lbStepPending = false;
  waitParked = false;
  // (CkLocRec::deviceRecvParked default-initializes to false; myRec is not
  // necessarily wired yet in this init path, so it is not touched here.)
  lbWaitEpoch = 0;

#if CMK_LBDB_ON
  // Join whatever step this PE has already been asked for rather than trying to
  // start it: a chare created (or migrated in) mid-step would otherwise call
  // atBarrier for an epoch whose local barrier has already fired.
  lbStepSeen = CkpvAccess(_lbStepRequested);
  if (_lb_args.metaLbOn())
  {
    // Deliberately not joining MetaBalancer's sample stream here. Joining means
    // recording which sample index this element will first contribute to, and
    // pre-crediting the open buckets it will therefore skip -- both of which are
    // only true of the instant they are done. Once elements can migrate while
    // their new PE is still sampling, which is what +LBAsync allows, that PE can
    // close buckets between this constructor and this element's first
    // AtSyncSample, and the recorded index is then already in the past. The join
    // happens on the first sample instead, in sampleMetaLBLoad, where the
    // instant it describes is the instant it is used.
    atsync_iteration = -1;
  }
#endif
}

CkMigratable::CkMigratable(void)
{
  DEBC((AA "In CkMigratable constructor\n" AB));
  commonInit();
}
CkMigratable::CkMigratable(CkMigrateMessage* m) : Chare(m) { commonInit(); }

int CkMigratable::ckGetChareType(void) const { return thisChareType; }

void CkMigratable::pup(PUP::er& p)
{
  DEBM((AA "In CkMigratable::pup %s\n" AB, idx2str(thisIndexMax)));
  Chare::pup(p);
  p | thisIndexMax;
  p | usesAtSync;
  p | can_reset;
  p | usesAutoMeasure;

#if CMK_LBDB_ON
  bool readyMigrate = false;
  if (p.isPacking())
    readyMigrate = myRec->isReadyMigrate();
  p | readyMigrate;
  if (p.isUnpacking())
    myRec->ReadyMigrate(readyMigrate);
#endif

  int epoch = -1;
  // Only pup epoch when migrating. Will result in problems if pup'd when
  // checkpointing since it will not match the value of the freshly inited barrier
  // upon restart
  if (usesAtSync && p.isMigration())
  {
    if (p.isPacking())
    {
      epoch = (*ldBarrierHandle)->epoch;
    }
    p | epoch;
    // +LBAsync wait state. An element can be parked in AtSyncWait() at the
    // moment the step it is waiting on moves it, and the destination has to be
    // able to finish that wait -- see lbCheckWaitRelease. Not checkpointed, for
    // the same reason the barrier epoch is not: it describes a step in flight.
    p | lbStepPending;
    p | waitParked;
    p | lbWaitEpoch;
    p | lbStepSeen;
    p | lbIterNo;
    p | lbStepBlocked;
  }

  if (p.isUnpacking())
    ckFinishConstruction(epoch);
}

void CkMigratable::ckDestroy(void) {}
void CkMigratable::ckAboutToMigrate(void) {}

#if CMK_LBDB_ON
void CkMigratable::ckSuspendBarrierForDeferredDestroy()
{
  // Remove this chare from the AtSync barrier so it is not woken by the next
  // ResumeClients. myRec stays alive; CkLocMgr still holds the chare and
  // deletes it when the migration ack arrives.
  if (usesAtSync && barrierRegistered)
  {
    myRec->getSyncBarrier()->removeClient(ldBarrierHandle);
    barrierRegistered = false;
  }
}
#endif

int CkMigratable::ckPrepareIntraProcessMigrate()
{
#if (CMK_CUDA || CMK_HIP) && CMK_GPU_COMM
  // Reallocates the element's device buffers just as the cross-PE path does,
  // so cached registrations for them have to go too.
  CkRdmaDeviceDropRegistrations(myRec);
#endif
  int epoch = -1;
  if (usesAtSync && barrierRegistered)
  {
    epoch = (*ldBarrierHandle)->epoch;
    myRec->getSyncBarrier()->removeClient(ldBarrierHandle);
    barrierRegistered = false;
  }
#if CMK_LBDB_ON
  // This element is leaving this PE's MetaBalancer sample stream without being
  // destroyed, and ~CkMigratable -- the only other place that decrements a
  // stream -- therefore never runs for this PE. Without this the stream goes on
  // waiting for a contribution that has become somebody else's, which surfaces
  // on the far side of the arithmetic as "Abort!!! Received more contribution".
  // Pairs with the credit the destination makes in sampleMetaLBLoad when it
  // sees the element arrive on a PE whose stream it has not credited.
  if (_lb_args.metaLbOn() && atsync_iteration != -1)
    myRec->getMetaBalancer()->AdjustCountForDeadContributor(atsync_iteration);
#endif
  myRec = nullptr;
  return epoch;
}

void CkMigratable::ckFinalizeIntraProcessMigrate(CkLocRec* newRec, int epoch)
{
  myRec = newRec;
  ckFinishConstruction(epoch);
}
void CkMigratable::ckJustMigrated(void) {}
void CkMigratable::ckJustRestored(void) {}

CkMigratable::~CkMigratable()
{
  DEBC((AA "In CkMigratable::~CkMigratable %s\n" AB, idx2str(thisIndexMax)));
#if CMK_OUT_OF_CORE
  isInCore = false;
  if (CkpvAccess(CkSaveRestorePrefetch))
    return; /* Just saving to disk--don't deregister anything. */
  /* We're really leaving or dying-- unregister from the ooc system*/
  if (prefetchObjID != -1)
  {
    CooDeregisterObject(prefetchObjID);
    prefetchObjID = -1;
  }
#endif
#if CMK_LBDB_ON
  if (barrierRegistered)
  {
    DEBL((AA "Removing barrier for element %s\n" AB, idx2str(thisIndexMax)));
    if (usesAtSync)
      myRec->getSyncBarrier()->removeClient(ldBarrierHandle);
  }

  // atsync_iteration == -1 means this element never joined the sample stream,
  // so no bucket is waiting on it and there is nothing to adjust.
  if (_lb_args.metaLbOn() && atsync_iteration != -1)
  {
    myRec->getMetaBalancer()->AdjustCountForDeadContributor(atsync_iteration);
  }
#endif
  myRec->destroy(); /* Attempt to delete myRec if it's no longer in use */
  // To detect use-after-delete
  thisIndexMax.nInts = 0;
  thisIndexMax.dimension = 0;
}

void CkMigratable::CkAbort(const char* format, ...) const
{
  char newmsg[256];
  va_list args;
  va_start(args, format);
  vsnprintf(newmsg, sizeof(newmsg), format, args);
  va_end(args);

  ::CkAbort("CkMigratable '%s' aborting: %s", _chareTable[thisChareType]->name, newmsg);
}

void CkMigratable::ResumeFromSync(void) {}

void CkMigratable::UserSetLBLoad()
{
  CkAbort("::UserSetLBLoad() not defined for this array element!\n");
}

#if CMK_LBDB_ON  // For load balancing:
// user can call this helper function to set obj load (for model-based lb)
void CkMigratable::setObjTime(double cputime) { 
  myRec->setObjTime(cputime); }
double CkMigratable::getObjTime() { return myRec->getObjTime(); }

void CkMigratable::setObjGPUTime(double gputime) {
  myRec->setObjGPUTime(gputime);
}
double CkMigratable::getObjGPUTime() { return myRec->getObjGPUTime(); }

#  if CMK_LB_USER_DATA
/**
 * Use this method to set user specified data to the lbdatabase.
 *
 * Eg usage:
 * In the application code,
 *   void *data = getObjUserData(CkpvAccess(_lb_obj_index));
 *   *(int *) data = val;
 *
 * In the loadbalancer or wherever this data is used do
 *   for (int i = 0; i < stats->n_objs; i++ ) {
 *     LDObjData &odata = stats->objData[i];
 *     int* udata = (int *) odata.getUserData(CkpvAccess(_lb_obj_index));
 *   }
 *
 * For a complete example look at tests/charm++/load_balancing/lb_userdata_test/
 */
void* CkMigratable::getObjUserData(int idx) { return myRec->getObjUserData(idx); }
#  endif

void CkMigratable::clearMetaLBData()
{
  //  if (can_reset) {
  local_state = OFF;
  // The GPU trigger's sample stream runs free across load balancing steps:
  // every PE completes every sample index in order, which is what keeps their
  // contributions to MetaBalancer's reduction lined up. Resetting the counter
  // here would make this chare re-contribute indices its PE has already closed.
  // prev_load still resets, so the sample that straddles ClearLoads measures
  // from zero rather than going negative.
  if (!_lb_args.metaLbGpuTrigger()) atsync_iteration = -1;
  prev_load = 0.0;
  can_reset = false;
  //  }
}

void CkMigratable::recvLBPeriod(void* data)
{
  if (_lb_args.metaLbGpuTrigger())
  {
    // The GPU trigger schedules on lbIterNo, not on the wall-time path's
    // extrapolated period, so none of the state machine below applies. This is
    // the agreed count arriving: a chare stopped at the tentative floor either
    // joins here, or is let go to keep iterating until it reaches the count.
    if (!lbStepBlocked) return;
    lbStepBlocked = false;
    const int agreed = *((int*)data);
    if (lbIterNo >= agreed)
    {
      lbJoinStep(1);
      // Under +LBAsync joining never stops a chare, so one that was stopped at
      // the tentative floor has to be handed back to the application here --
      // its AtSyncStart() already returned, so nothing else will do it.
      if (_lb_args.lbAsync()) ResumeFromSync();
    }
    else
    {
      ResumeFromSync();
    }
    return;
  }
  if (atsync_iteration < 0)
  {
    return;
  }
  int lb_period = *((int*)data);
  DEBAD(("\t[obj %s] Received the LB Period %d current iter %d state %d on PE %d\n",
         idx2str(thisIndexMax), lb_period, atsync_iteration, local_state, CkMyPe()));

  bool is_tentative;
  if (local_state == LOAD_BALANCE)
  {
    CkAssert(lb_period == myRec->getMetaBalancer()->getPredictedLBPeriod(is_tentative));
    return;
  }

  if (local_state == PAUSE)
  {
    if (atsync_iteration < lb_period)
    {
      local_state = DECIDED;
      ResumeFromSync();
      return;
    }
    local_state = LOAD_BALANCE;

    can_reset = true;
    return;
  }
  local_state = DECIDED;
}

void CkMigratable::metaLBCallLB()
{
  if (usesAtSync)
    myRec->getSyncBarrier()->atBarrier(ldBarrierHandle);
}

void CkMigratable::ckFinishConstruction(int epoch)
{
  //	if ((!usesAtSync) || barrierRegistered) return;
  if (usesAtSync && _lb_args.lbperiod() != -1.0)
    CkAbort("You must use AtSync or Periodic LB separately!\n");

  myRec->setMeasure(usesAutoMeasure);
  if (barrierRegistered)
    return;
  DEBL((AA "Registering barrier client for %s\n" AB, idx2str(thisIndexMax)));
  if (usesAtSync)
  {
    ldBarrierHandle = myRec->getSyncBarrier()->addClient(
        this, [=]() { this->ResumeFromSyncHelper(); }, epoch);
  }
  barrierRegistered = true;
}

void CkMigratable::AtSync(int waitForMigration)
{
  if (!usesAtSync)
    CkAbort(
        "You must set usesAtSync=true in your array element constructor to use "
        "AtSync!\n");
  // Only actually call AtSync when a receiver exists, otherwise skip it and
  // directly call ResumeFromSync
  if (!myRec->getSyncBarrier()->hasReceivers())
  {
    ResumeFromSync();
    return;
  }
  myRec->AsyncMigrate(!waitForMigration);
  if (waitForMigration)
    ReadyMigrate(true);
  ckFinishConstruction();
  DEBL((AA "Element %s going to sync\n" AB, idx2str(thisIndexMax)));
  recordLBSizes(true);

  if (!_lb_args.metaLbOn())
  {
    myRec->getSyncBarrier()->atBarrier(ldBarrierHandle);
    return;
  }

  // When MetaBalancer is turned on

  sampleMetaLBLoad();

  bool is_tentative;
  if (atsync_iteration < myRec->getMetaBalancer()->getPredictedLBPeriod(is_tentative))
  {
    ResumeFromSync();
  }
  else if (is_tentative)
  {
    local_state = PAUSE;
  }
  else if (local_state == DECIDED)
  {
    DEBAD(("[%d:%s] Went to load balance iter %d\n", CkMyPe(), idx2str(thisIndexMax),
           atsync_iteration));
    local_state = LOAD_BALANCE;
    can_reset = true;
  }
  else
  {
    DEBAD(("[%d:%s] Went to pause state iter %d\n", CkMyPe(), idx2str(thisIndexMax),
           atsync_iteration));
    local_state = PAUSE;
  }
}

// Measure what the load balancer needs to know about this chare: its load, its
// serialized size, and -- only when a step is actually starting -- the device
// footprint the memory contract prices migration with. A sample skips that last
// part: it is a second full pup traversal, and nothing reads the result until a
// step runs.
void CkMigratable::recordLBSizes(bool forStep)
{
  // model-based load balancing, ask user to provide cpu load
  if (usesAutoMeasure == false)
    UserSetLBLoad();

  #ifdef CMK_CUDA
  if (forStep) {
  PUP::sizer ps(PUP::er::IS_MIGRATION);
  this->virtual_pup(ps);
  // printf("[%d] gpu pup size %ld\n",CkMyPe(), ps.gpu_size() );
  setGPUPupSize(ps.gpu_size());
  #if CMK_LB_USER_DATA
  // Per-chare device footprint, written into the LB user-data slot the
  // memory-aware strategies read. Floored at the serialized size so an
  // unattributed footprint (allocations outside entry methods, arena
  // storage) can never read as "free to move".
  {
    const int udIdx = CkpvAccess(_lb_obj_index);
    void* ud = (udIdx >= 0) ? getObjUserData(udIdx) : NULL;
    size_t footprint = 0;
    if (ud != NULL) {
      footprint = hapiCurrentObjectFootprint();
      if (footprint < ps.gpu_size()) footprint = ps.gpu_size();
      *(size_t*)ud = footprint;
    }
    if (getenv("CHARM_DEBUG_GPU_FOOTPRINT") != NULL)
      CkPrintf("[%d] LB footprint: idx=%d ud=%p fp=%zu pup=%zu\n",
               CkMyPe(), udIdx, ud, footprint, ps.gpu_size());
  }
  #endif
  }
  #endif
  if (_lb_psizer_on || _lb_args.metaLbOn())
  {
    PUP::sizer ps(PUP::er::IS_MIGRATION);
    this->virtual_pup(ps);
    if (_lb_psizer_on)
    {
      setPupSize(ps.size());
    }
    //TODO: check if this is correct after gpuPUP size
    if (_lb_args.metaLbOn())
      myRec->getMetaBalancer()->SetCharePupSize(ps.size());
  }
}

// Fold this chare's load since its last sample into the local MetaBalancer.
// Once every chare on this PE has done so for a sample index, the PE contributes
// it to a background reduction. Nothing here is collective and nothing blocks.
void CkMigratable::sampleMetaLBLoad()
{
  if (atsync_iteration == -1)
  {
    can_reset = false;
    local_state = OFF;
    prev_load = 0.0;
    // Join the stream where this PE is right now. AdjustCountForNewContributor
    // credits the buckets still open here, which are exactly the ones this
    // element is skipping past -- so they can complete without it, and the one
    // it is about to open is the first it owes.
    atsync_iteration = myRec->getMetaBalancer()->get_iteration();
    myRec->getMetaBalancer()->AdjustCountForNewContributor(atsync_iteration);
    metaLBStreamJoinedPe = CkMyPe();
  }
  else if (metaLBStreamJoinedPe != CkMyPe())
  {
    // Arrived from another PE carrying that PE's sample index. Streams advance
    // per PE, so the index it brings can already be behind this one -- and then
    // the credit below, which only covers finished+1 through atsync_iteration,
    // cannot reach the bucket it is about to contribute to. It contributes to a
    // bucket this PE has already closed and trips the guard further down.
    // Join where this PE actually is, exactly as a first-time contributor does.
    const int here = myRec->getMetaBalancer()->get_iteration();
    if (atsync_iteration < here) atsync_iteration = here;
    // Arrived by migration, carrying its own atsync_iteration, so the branch
    // above did not run. This PE's open buckets still have to be credited for a
    // contributor they were not counting, exactly as for a new chare -- without
    // it this PE eventually sees more contributions to an index than it has
    // objects, and aborts.
    myRec->getMetaBalancer()->AdjustCountForNewContributor(atsync_iteration);
    metaLBStreamJoinedPe = CkMyPe();
  }

  atsync_iteration++;
  // CkPrintf("[pe %s] atsync_iter %d && predicted period %d state: %d\n",
  //    idx2str(thisIndexMax), atsync_iteration,
  //    myRec->getMetaBalancer()->getPredictedLBPeriod(), local_state);
  double tmp = prev_load;
  prev_load = myRec->getObjTime();
  double current_load = prev_load - tmp;

  // If the load for the chares are based on certain model, then set the
  // current_load to be whatever is the obj load.
  if (!usesAutoMeasure)
  {
    current_load = myRec->getObjTime();
  }

  // A step's ClearLoads can land inside a sample window, restarting the object
  // timer below the reading this chare last took.
  if (current_load < 0.0) current_load = 0.0;

  if (atsync_iteration <= myRec->getMetaBalancer()->get_finished_iteration())
  {
    CkPrintf("[%d:%s] Error!! Contributing to iter %d < current iter %d\n", CkMyPe(),
             idx2str(thisIndexMax), atsync_iteration,
             myRec->getMetaBalancer()->get_finished_iteration());
    CkAbort("Not contributing to the right iteration\n");
  }

  if (atsync_iteration != 0)
  {
    myRec->getMetaBalancer()->AddLoad(atsync_iteration, current_load);
  }
}

// Contribute load to MetaBalancer without going anywhere near the barrier. Call
// this every few iterations: it is the expensive half (on CUDA it flushes and
// normalizes the GPU counters once per sample per process), and MetaBalancer
// only needs a periodic reading to spot an imbalance.
void CkMigratable::AtSyncSample()
{
  if (!usesAtSync)
    CkAbort(
        "You must set usesAtSync=true in your array element constructor to use "
        "AtSyncSample!\n");
  // No balancer is listening, or MetaBalancer is off and nothing consumes the
  // sample. Either way this is a no-op, so applications can call it blind.
  if (!_lb_args.metaLbOn() || !myRec->getSyncBarrier()->hasReceivers()) return;

  ckFinishConstruction();
  recordLBSizes(false);
  sampleMetaLBLoad();
  // The decision is made from this sample, so this is where the measurement
  // window closes. It opened when AtSyncWait() released this chare, which the
  // application places a few iterations back -- so what the strategy sees is a
  // short, recent window, measured entirely after the previous step's
  // migrations settled and ending at the decision itself.
  LBTurnInstrumentOff();
}

bool CkMigratable::AtSyncPending() const
{
  if (!usesAtSync || !myRec->getSyncBarrier()->hasReceivers()) return false;
  // Without MetaBalancer every AtSyncStart starts a step, so one is always due.
  if (!_lb_args.metaLbOn()) return true;
  // A step is agreed, and this chare's next AtSyncStart is the one it was
  // agreed for. Both halves matter: a step can be requested many iterations
  // before the count it is to be joined at.
  if (CkpvAccess(_lbStepRequested) <= lbStepSeen) return false;
  const int agreed = CkpvAccess(_lbStepPeriod);
  return agreed > 0 && lbIterNo + 1 >= agreed;
}

// Start a load balancing step if one has been requested since this chare last
// joined one. Cheap enough to call every iteration -- which is the point: a
// step begins one iteration after MetaBalancer sees the imbalance, however
// rarely AtSyncSample runs.
//
// Without +LBAsync this behaves like AtSync: the element joins the barrier and
// waits for ResumeFromSync, and when no step is due it resumes inline. With
// +LBAsync it never resumes -- it returns to its caller, which keeps running
// while the strategy and the migrations proceed, and AtSyncWait() is where the
// element finds out the step is over.
CkMigratable::AtSyncStatus CkMigratable::AtSyncStart(int waitForMigration)
{
  if (!usesAtSync)
    CkAbort(
        "You must set usesAtSync=true in your array element constructor to use "
        "AtSyncStart!\n");
  const bool async = _lb_args.lbAsync();

  if (!myRec->getSyncBarrier()->hasReceivers())
  {
    if (!async) ResumeFromSync();
    return AtSyncStatus::Continue;
  }

  // One step at a time, per element. Joining again would pass the barrier
  // twice for one element and could fire the next step's barrier before this
  // step's migrations had even been released.
  if (lbStepPending)
    CkAbort(
        "AtSyncStart() called while the load balancing step this element started "
        "is still in flight. Every AtSyncStart() must be followed by an "
        "AtSyncWait() before the next one.\n");

  // With MetaBalancer off there is nothing to decide for us, so every call
  // starts a step and the application's own cadence is the trigger.
  if (_lb_args.metaLbOn())
  {
    // The application calls this on a fixed cadence, so this counter is the
    // clock every chare agrees on. It has to advance on every call, including
    // the calls with no step to start, or the counts drift apart.
    ++lbIterNo;
    myRec->getMetaBalancer()->reportStartNo(lbIterNo);

    if (CkpvAccess(_lbStepRequested) <= lbStepSeen)
    {
      if (!async) ResumeFromSync();
      return AtSyncStatus::Continue;
    }

    const int agreed = CkpvAccess(_lbStepPeriod);
    if (agreed == 0)
    {
      // A step has been called for but its count is still being agreed. Run up
      // to the tentative floor and stop there. Stopping is the whole mechanism:
      // it bounds how far any chare can get before the count is settled, so the
      // agreed count is one no chare has passed. Relying instead on the final
      // broadcast outrunning the application would make correctness a race.
      if (lbIterNo < CkpvAccess(_lbStepTentative))
      {
        if (!async) ResumeFromSync();
        return AtSyncStatus::Continue;
      }
      lbStepBlocked = true;
      if (migDbg())
        CkPrintf("[LBSTEP] pe=%d elem=%s BLOCK iter=%d tentative=%d\n", CkMyPe(),
                 idx2str(thisIndexMax), lbIterNo, CkpvAccess(_lbStepTentative));
      return AtSyncStatus::Blocked;
    }

    if (lbIterNo < agreed)
    {
      if (!async) ResumeFromSync();
      return AtSyncStatus::Continue;
    }
  }

  lbJoinStep(waitForMigration);
  return AtSyncStatus::Started;
}

// The tail of a step join, shared by AtSyncStart and the release that happens
// when a blocked chare learns the agreed count.
void CkMigratable::lbJoinStep(int waitForMigration)
{
  if (_lb_args.metaLbOn()) lbStepSeen = CkpvAccess(_lbStepRequested);

  myRec->AsyncMigrate(!waitForMigration);
  // Migratable straight away, in both modes. An element that keeps running
  // under +LBAsync can still be moved at any entry-method boundary, because
  // making its own state safe to pack is its pup's job -- an element with
  // device state settles its streams on the packing pass, which is the only
  // code that knows which streams those are. Holding the move until AtSyncWait
  // instead would mean an element never overlaps its own migration with its own
  // iterations, which is most of the point of the split.
  if (waitForMigration)
    ReadyMigrate(true);
  ckFinishConstruction();
  DEBL((AA "Element %s starting LB step %d\n" AB, idx2str(thisIndexMax), lbStepSeen));
  recordLBSizes(true);

  local_state = LOAD_BALANCE;
  can_reset = true;
  lbStepPending = true;
  lbWaitEpoch = myRec->getLBMgr()->resumeEpoch();
  if (migDbg())
    CkPrintf("[LBSTEP] pe=%d elem=%s JOIN seq=%d iter=%d epoch=%d\n", CkMyPe(),
             idx2str(thisIndexMax), lbStepSeen, lbIterNo, lbWaitEpoch);
  myRec->getSyncBarrier()->atBarrier(ldBarrierHandle);
}

// The blocking half of the split. See the header for the contract.
void CkMigratable::AtSyncWait()
{
  if (!usesAtSync)
    CkAbort(
        "You must set usesAtSync=true in your array element constructor to use "
        "AtSyncWait!\n");
  // Without +LBAsync the element is already stopped waiting for the barrier's
  // resume, and adding one here would resume it twice.
  if (!_lb_args.lbAsync()) return;

  if (!lbStepPending)
  {
    // The step already finished, so the wait is where the window opens.
    if (_lb_args.metaLbOn()) LBTurnInstrumentOn();
    ResumeFromSync();
    return;
  }
  waitParked = true;
  myRec->deviceRecvParked = true;
  if (migDbg())
    CkPrintf("[PARK %d] elem %s pending=%d count=%d\n", CkMyPe(),
             idx2str(thisIndexMax), myRec->pendingMigrateTo,
             myRec->outstandingDeviceSends);
  // A migration decided while this element was still running was deferred to
  // this park. Enqueued, not driven inline: this is the element's own entry
  // method, and emigrate deletes the object under it.
  if (myRec->pendingMigrateTo != -1 && myRec->outstandingDeviceSends == 0)
  {
    const int toPe = myRec->pendingMigrateTo;
    myRec->pendingMigrateTo = -1;
    DeferredMigrateMsg* m =
        (DeferredMigrateMsg*)CmiAlloc(sizeof(DeferredMigrateMsg));
    m->mgrGid = myRec->getLocMgr()->ckGetGroupID();
    m->id = myRec->getID();
    m->toPe = toPe;
    CmiSetHandler(m, _deferredMigrateHandlerIdx);
    CmiPushPE(CmiMyRank(), m);
  }
}

// An element that parked in AtSyncWait() and was then moved by the very step it
// was waiting on can land on its destination after that PE resumed its clients,
// so nothing there would ever call it back. Under +LBSyncResume, which +LBAsync
// forces on, the resume epoch advances globally in step, so comparing the
// destination's against the one the element parked at settles it.
void CkMigratable::lbCheckWaitRelease()
{
  if (!_lb_args.lbAsync() || !lbStepPending) return;
  if (myRec->getLBMgr()->resumeEpoch() <= lbWaitEpoch) return;

  lbStepPending = false;
  if (!waitParked) return;
  waitParked = false;
  myRec->deviceRecvParked = false;
  CkDeviceRecvAdmissionReplay(myRec->getID());
  DEBL((AA "Element %s released after migrating into a resumed PE\n" AB,
        idx2str(thisIndexMax)));
  if (_lb_args.metaLbOn()) LBTurnInstrumentOn();
  ResumeFromSync();
}

void CkMigratable::ReadyMigrate(bool ready) { myRec->ReadyMigrate(ready); }

void CkMigratable::ResumeFromSyncHelper()
{
  DEBL((AA "Element %s resuming from sync\n" AB, idx2str(thisIndexMax)));

  if (_lb_args.metaLbOn())
  {
    clearMetaLBData();
  }

  // The step this element started is over, in either mode -- AtSyncStart()
  // refuses to begin another while this is set.
  if (migDbg())
    CkPrintf("[LBSTEP] pe=%d elem=%s RESUME seen=%d epoch=%d\n", CkMyPe(),
             idx2str(thisIndexMax), lbStepSeen, myRec->getLBMgr()->resumeEpoch());
  lbStepPending = false;

  if (_lb_args.lbAsync())
  {
    // Starting the step never stopped the element, so there is nothing to
    // resume unless it went on to park in AtSyncWait().
    if (!waitParked) return;
    waitParked = false;
    myRec->deviceRecvParked = false;
    CkDeviceRecvAdmissionReplay(myRec->getID());
  }
  // Released from the wait -- open the measurement window. Under +LBAsync this
  // is AtSyncWait() returning, which the application places a few iterations
  // before its next AtSyncSample(); without it the step's own resume is the
  // same moment. Either way nothing between the step and here is measured.
  if (_lb_args.metaLbOn()) LBTurnInstrumentOn();

  CkLocMgr* localLocMgr = myRec->getLocMgr();
  auto iter = localLocMgr->bufferedActiveRgetMsgs.find(ckGetID());
  if (iter != localLocMgr->bufferedActiveRgetMsgs.end())
  {
    localLocMgr->toBeResumeFromSynced.emplace(ckGetID(), this);
  }
  else
  {
    ResumeFromSync();
  }
}

void CkMigratable::setMigratable(int migratable) { myRec->setMigratable(migratable); }

void CkMigratable::setPupSize(size_t obj_pup_size) { myRec->setPupSize(obj_pup_size); }

void CkMigratable::setGPUPupSize(size_t obj_gpu_pup_size) { myRec->setGPUPupSize(obj_gpu_pup_size); }


void CkMigratable::CkAddThreadListeners(CthThread tid, void* msg)
{
  Chare::CkAddThreadListeners(tid, msg);  // for trace
  CthSetThreadID(tid, thisIndexMax.data()[0], thisIndexMax.data()[1],
                 thisIndexMax.data()[2]);
}
#else
void CkMigratable::setObjTime(double cputime) {}
double CkMigratable::getObjTime() { return 0.0; }
void CkMigratable::setObjGPUTime(double gputime) {}
double CkMigratable::getObjGPUTime() { return 0.0; }

#  if CMK_LB_USER_DATA
void* CkMigratable::getObjUserData(int idx) { return NULL; }
#  endif

/* no load balancer: need dummy implementations to prevent link error */
void CkMigratable::CkAddThreadListeners(CthThread tid, void* msg) {}
#endif

/************************** Location Records: *********************************/

/*----------------- Local:
Matches up the array index with the local index, an
interfaces with the load balancer on behalf of the
represented array elements.
*/
CkLocRec::CkLocRec(CkLocMgr* mgr, bool fromMigration, bool ignoreArrival,
                   const CkArrayIndex& idx_, CmiUInt8 id_)
    : myLocMgr(mgr), idx(idx_), id(id_), deletedMarker(NULL), running(false)
{
#if CMK_LBDB_ON
  DEBL((AA "Registering element %s with load balancer\n" AB, idx2str(idx)));
  nextPe = -1;
  asyncMigrate = false;
  readyMigrate = true;
  enable_measure = true;
  syncBarrier = CkSyncBarrier::object();
  lbmgr = mgr->getLBMgr();
  if (_lb_args.metaLbOn())
    the_metalb = mgr->getMetaBalancer();
#  if CMK_GLOBAL_LOCATION_UPDATE
  id_ = ck::ObjID(mgr->getGroupID(), id_).getID();
#  endif
  ldHandle = lbmgr->RegisterObj(mgr->getOMHandle(), id_, (void*)this, 1);
  if (fromMigration)
  {
    DEBL((AA "Element %s migrated in\n" AB, idx2str(idx)));
    // An asynchronous arrival is one the load balancer does not hold its
    // migration barrier for, but it must still be counted. CentralLB tallies
    // these separately in future_migrates_completed, and CheckMigrationComplete
    // withholds its second token -- and with it DoneRegisteringObjects, and so
    // the next LB step -- until every one of them has landed. Skipping the call
    // entirely left future_migrates_completed at zero against a non-zero
    // future_migrates_expected, and the step after an async migration hung.
    lbmgr->Migrated(ldHandle, !ignoreArrival);
  }
#endif
}
CkLocRec::~CkLocRec()
{
  if (deletedMarker != NULL)
    *deletedMarker = true;
#if CMK_LBDB_ON
  stopTiming();
  DEBL((AA "Unregistering element %s from load balancer\n" AB, idx2str(idx)));
  lbmgr->UnregisterObj(ldHandle);
#endif
}
void CkLocRec::migrateMe(int toPe)  // Leaving this processor
{
  // This will pack us up, send us off, and delete us
  //	printf("[%d] migrating migrateMe to %d \n",CkMyPe(),toPe);
  myLocMgr->emigrate(this, toPe);
}

#if CMK_LBDB_ON
CkpvDeclare(CkLocRec*, _currentLocRec);


// A device receive deferred for correction resumes against the destination
// buffer captured before the request round trip. If the element migrates in
// that window its buffers are freed, and the correction writes into memory that
// has since been handed to another chare -- silent corruption -- while the
// message is enqueued to the PE the element has left. Count these against the
// element exactly as an outstanding send is counted, so emigrate stands down
// until the correction lands.
void CkNoteDeviceRecvDeferred(CkGroupID aid, CmiUInt8 id)
{
  if (aid.idx == -1) return;   // not an array element; nothing to stand down
  CkArray* arr = (CkArray*)CkLocalBranch(aid);
  CkLocMgr* mgr = arr ? arr->getLocMgr() : NULL;
  // The rec table is keyed by the bare element id; `id` arrives as the
  // envelope's packed ObjID (collection|home|element), which never matches it
  // as-is. Looking up the packed value here made this stand-down a silent
  // no-op for every element -- migration proceeded with corrections in
  // flight, and the correction then wrote into the departed element's freed
  // buffers.
  CkLocRec* rec = mgr ? mgr->elementNrec(ck::ObjID(id).getElementID()) : NULL;
  if (rec == NULL) return;   // already gone; nothing left here to hold
  rec->noteDeviceSendPosted();
}

void CkNoteDeviceRecvComplete(CkGroupID aid, CmiUInt8 id)
{
  if (aid.idx == -1) return;   // not an array element; nothing to stand down
  CkArray* arr = (CkArray*)CkLocalBranch(aid);
  CkLocMgr* mgr = arr ? arr->getLocMgr() : NULL;
  CkLocRec* rec = mgr ? mgr->elementNrec(ck::ObjID(id).getElementID()) : NULL;
  if (rec == NULL) return;   // departed; the stand-down it held died with it
  rec->noteDeviceSendDone();
}

// Same resolution, exported as a lookup: the receive path of the device
// registration cache (ckrdmadevice.C) attributes a destination-buffer
// registration to the element that posted it, so the entry is retired when
// that element migrates and its buffers are freed.
CkLocRec* CkFindDeviceRecvElement(CkGroupID aid, CmiUInt8 id)
{
  if (aid.idx == -1) return NULL;
  CkArray* arr = (CkArray*)CkLocalBranch(aid);
  CkLocMgr* mgr = arr ? arr->getLocMgr() : NULL;
  return mgr ? mgr->elementNrec(ck::ObjID(id).getElementID()) : NULL;
}


// Releasing the last outstanding send is what lets a migration that was waiting
// on it proceed. Deferring rather than blocking matters: an inter-node send
// completes only when the receiver acknowledges, so spinning here would deadlock
// against a receiver that is itself mid-migration.
// (DeferredMigrateMsg and its handler index are declared near the top of
// this file; the handler body lives beside noteDeviceSendDone below.)

extern "C" void _deferredMigrateHandler(void* arg)
{
  DeferredMigrateMsg* m = (DeferredMigrateMsg*)arg;
  CkLocMgr* mgr = (CkLocMgr*)CkLocalBranch(m->mgrGid);
  CkLocRec* rec = mgr ? mgr->elementNrec(m->id) : NULL;
  if (rec != NULL) {
    bool movable = (rec->outstandingDeviceSends == 0);
#if CMK_LBDB_ON
    if (_lb_args.lbAsync() && !rec->deviceRecvParked) movable = false;
#endif
    if (getenv("CHARM_DEBUG_MIGRATE"))
      CmiPrintf("[KICK %d] id=%llu movable=%d toPe=%d\n", CkMyPe(),
                (unsigned long long)m->id, (int)movable, m->toPe);
    if (movable) {
      mgr->emigrate(rec, m->toPe);  // does not return to this record
    } else if (rec->pendingMigrateTo == -1) {
      // New transfers started between the zero crossing that enqueued this
      // kick and now. Re-arm rather than drop: the next zero crossing kicks
      // again, and dropping here would lose the migration outright -- the
      // balancer's resume barrier then waits forever for a move that no one
      // is going to make.
      rec->pendingMigrateTo = m->toPe;
    }
  }
  // An element that moved or died since the enqueue made this moot.
  CmiFree(m);
}

void CkLocRec::noteDeviceSendDone()
{
  if (outstandingDeviceSends > 0) outstandingDeviceSends--;
  if (outstandingDeviceSends == 0 && pendingMigrateTo != -1)
  {
    const int toPe = pendingMigrateTo;
    pendingMigrateTo = -1;
    // Enqueued, never called inline. The release that brought the count to
    // zero can run inside the very handler that has just enqueued this
    // element's completion message: migrating here packs the element before
    // that message is consumed, and the message is then forwarded to the new
    // process carrying pointers into this one. Going through the scheduler
    // puts the migration behind everything already enqueued -- including that
    // message -- at the cost of one local hop.
    DeferredMigrateMsg* m = (DeferredMigrateMsg*)CmiAlloc(sizeof(DeferredMigrateMsg));
    m->mgrGid = myLocMgr->ckGetGroupID();
    m->id = getID();
    m->toPe = toPe;
    CmiSetHandler(m, _deferredMigrateHandlerIdx);
    CmiPushPE(CmiMyRank(), m);
  }
}

void CkLocRec::startTiming(int ignore_running)
{
  if (!ignore_running)
    running = true;
  CkpvAccess(_currentLocRec) = this;
  DEBL((AA "Start timing for %s at %.3fs {\n" AB, idx2str(idx), CkWallTimer()));
  if (enable_measure)
    lbmgr->ObjectStart(ldHandle);
}
void CkLocRec::stopTiming(int ignore_running)
{
  DEBL((AA "} Stop timing for %s at %.3fs\n" AB, idx2str(idx), CkWallTimer()));
  if ((ignore_running || running) && enable_measure)
    lbmgr->ObjectStop(ldHandle);
  if (!ignore_running) {
    running = false;
    if (CkpvAccess(_currentLocRec) == this) CkpvAccess(_currentLocRec) = NULL;
  }
}
void CkLocRec::setObjTime(double cputime) { 
  lbmgr->EstObjLoad(ldHandle, cputime); }
double CkLocRec::getObjTime()
{
  LBRealType walltime, cputime;
  lbmgr->GetObjLoad(ldHandle, walltime, cputime);
  return walltime;
}
void CkLocRec::setObjGPUTime(double gputime) {
  lbmgr->EstObjGPULoad(ldHandle, gputime);
}
double CkLocRec::getObjGPUTime()
{
  LBRealType gputime;
  lbmgr->GetObjGPULoad(ldHandle, gputime);
  return gputime;
}
#  if CMK_LB_USER_DATA
void* CkLocRec::getObjUserData(int idx) { return lbmgr->GetDBObjUserData(ldHandle, idx); }
#  endif
#endif

// Attempt to destroy this record. If the location manager is done with the
// record (because all array elements were destroyed) then it will be deleted.
void CkLocRec::destroy(void) { myLocMgr->reclaim(this); }

/****************************************************************************/

bool CkLocRec::invokeEntry(CkMigratable* obj, void* msg, int epIdx, bool doFree)
{
  DEBS((AA "   Invoking entry %d on element %s\n" AB, epIdx, idx2str(idx)));
  bool isDeleted = false;  // Enables us to detect deletion during processing
  deletedMarker = &isDeleted;

#if CMK_TRACE_ENABLED
  if (msg)
  { /* Tracing: */
    envelope* env = UsrToEnv(msg);
    //	CkPrintf("ckLocation.C beginExecuteDetailed %d %d
    //\n",env->getEvent(),env->getsetArraySrcPe());
    if (_entryTable[epIdx]->traceEnabled)
    {
      CmiObjId projID = idx.getProjectionID();
      _TRACE_BEGIN_EXECUTE_DETAILED(env->getEvent(), ForChareMsg, epIdx, env->getSrcPe(),
                                    env->getTotalsize(), &projID, obj);
      if (_entryTable[epIdx]->appWork)
        _TRACE_BEGIN_APPWORK();
    }
  }
#endif

  if (doFree)
    CkDeliverMessageFree(epIdx, msg, obj);
  else /* !doFree */
    CkDeliverMessageReadonly(epIdx, msg, obj);

#if CMK_TRACE_ENABLED
  if (msg)
  { /* Tracing: */
    if (_entryTable[epIdx]->traceEnabled)
    {
      if (_entryTable[epIdx]->appWork)
        _TRACE_END_APPWORK();
      _TRACE_END_EXECUTE();
    }
  }
#endif
#if CMK_LBDB_ON
  if (!isDeleted)
    checkBufferedMigration();  // check if should migrate
#endif
  if (isDeleted)
    return false;  // We were deleted
  deletedMarker = NULL;
  return true;
}

#if CMK_LBDB_ON

void CkLocRec::staticMetaLBResumeWaitingChares(LDObjHandle h, int lb_ideal_period)
{
  CkLocRec* el = (CkLocRec*)(LBManager::Object()->GetObjUserData(h));
  DEBL((AA "MetaBalancer wants to resume waiting chare %s\n" AB, idx2str(el->idx)));
  el->myLocMgr->informLBPeriod(el, lb_ideal_period);
}

void CkLocRec::staticMetaLBCallLBOnChares(LDObjHandle h)
{
  CkLocRec* el = (CkLocRec*)(LBManager::Object()->GetObjUserData(h));
  DEBL((AA "MetaBalancer wants to call LoadBalance on chare %s\n" AB, idx2str(el->idx)));
  el->myLocMgr->metaLBCallLB(el);
}

void CkLocRec::staticMigrate(LDObjHandle h, int dest)
{
  CkLocRec* el = (CkLocRec*)(LBManager::Object()->GetObjUserData(h));
  DEBL((AA "Load balancer wants to migrate %s to %d\n" AB, idx2str(el->idx), dest));
  el->recvMigrate(dest);
}

void CkLocRec::recvMigrate(int toPe)
{
  // we are in the mode of delaying actual migration
  // till readyMigrate()
  if (readyMigrate)
  {
    migrateMe(toPe);
  }
  else
    nextPe = toPe;
}

void CkLocRec::AsyncMigrate(bool use)
{
  asyncMigrate = use;
  lbmgr->UseAsyncMigrate(ldHandle, use);
}

bool CkLocRec::checkBufferedMigration()
{
  // we don't migrate in user's code when calling ReadyMigrate(true)
  // we postphone the action to here until we exit from the user code.
  if (readyMigrate && nextPe != -1)
  {
    int toPe = nextPe;
    nextPe = -1;
    // don't migrate inside the object call
    migrateMe(toPe);
    // don't do anything
    return true;
  }
  return false;
}

int CkLocRec::MigrateToPe()
{
  int pe = nextPe;
  nextPe = -1;
  return pe;
}

void CkLocRec::setMigratable(int migratable)
{
  if (migratable)
    lbmgr->Migratable(ldHandle);
  else
    lbmgr->NonMigratable(ldHandle);
}

void CkLocRec::setPupSize(size_t obj_pup_size)
{
  lbmgr->setPupSize(ldHandle, obj_pup_size);
}

void CkLocRec::setGPUPupSize(size_t obj_gpu_pup_size)
{
  lbmgr->setGPUPupSize(ldHandle, obj_gpu_pup_size);
}

#endif

// Call ckDestroy for each record, which deletes the record, and ~CkLocRec()
// removes it from the hash table, which would invalidate an iterator.
void CkLocMgr::flushLocalRecs(void)
{
  while (hash.size())
  {
    CkLocRec* rec = hash.begin()->second;
    callMethod(rec, &CkMigratable::ckDestroy);
  }
}

// All records are local records after the 64bit ID update
void CkLocMgr::flushAllRecs(void) { flushLocalRecs(); }

/*************************** LocCache **************************/
const CkLocEntry CkLocEntry::nullEntry = CkLocEntry();
void CkLocCache::pup(PUP::er& p)
{
#if __FAULT__
  if (!p.isUnpacking())
  {
    /**
     * pack the indexes of elements which have their homes on this processor
     * but dont exist on it.. needed for broadcast after a restart
     * indexes of local elements dont need to be packed since they will be
     * recreated later anyway
     */
    std::vector<CkLocEntry> entries;
    for (const auto& itr : locMap)
    {
      if (homePe(itr.first) == CmiMyPe() && itr.second.pe != CmiMyPe())
      {
        entries.push_back(itr.second);
      }
    }

    int count = entries.size();
    p | count;
    for (int i = 0; i < count; i++)
    {
      p | entries[i];
    }
  }
  else
  {
    int count;
    p | count;
    for (int i = 0; i < count; i++)
    {
      CkLocEntry e;
      p | e;
      e.epoch = 0;
      updateLocation(e);
      if (homePe(e.id) != CkMyPe())
      {
        thisProxy[homePe(e.id)].updateLocation(e);
      }
      CkAssert(getPe(e.id) == e.pe);
    }
  }
#endif
}

// CHARM_LB_MIGSTATS: how much wall time each migration path actually consumes,
// and how many objects take it. Separates the live-pointer handoff used inside a
// process from the pack/unpack path used across processes.
namespace {
struct MigStats {
  std::atomic<double> intraSecs{0.0}, packedSecs{0.0};
  std::atomic<long> intraN{0}, packedN{0};
  ~MigStats() {
    if (intraN.load() + packedN.load() == 0) return;
    fprintf(stderr, "[mig-stats] pid=%d intraproc n=%ld time=%.3fs | packed n=%ld time=%.3fs\n",
            (int)getpid(), intraN.load(), intraSecs.load(),
            packedN.load(), packedSecs.load());
    fflush(stderr);
  }
};
MigStats g_mig;
inline bool migStatsOn() {
  static const bool on = (getenv("CHARM_LB_MIGSTATS") != nullptr);
  return on;
}
struct MigTimer {
  double t0; bool on; std::atomic<double>* acc; std::atomic<long>* cnt;
  MigTimer(std::atomic<double>* a, std::atomic<long>* c)
      : t0(0), on(migStatsOn()), acc(a), cnt(c) { if (on) t0 = CkWallTimer(); }
  ~MigTimer() {
    if (!on) return;
    double d = CkWallTimer() - t0, cur = acc->load();
    while (!acc->compare_exchange_weak(cur, cur + d)) {}
    cnt->fetch_add(1, std::memory_order_relaxed);
  }
};
}  // namespace

void ckLocNoteWhichPeMiss(bool noId)
{
  static const bool on = (getenv("CHARM_ZC_STATS") != nullptr);
  if (!on) return;
  static std::atomic<long> noIdCount{0}, noLocCount{0};
  static struct Reporter {
    ~Reporter() {
      const long a = noIdCount.load(), b = noLocCount.load();
      if (a + b == 0) return;
      fprintf(stderr, "[whichPe-miss] pid=%d no_id=%ld no_location=%ld\n",
              (int)getpid(), a, b);
      fflush(stderr);
    }
  } reporter;
  if (noId) noIdCount.fetch_add(1, std::memory_order_relaxed);
  else noLocCount.fetch_add(1, std::memory_order_relaxed);
}

void CkLocCache::requestLocation(CmiUInt8 id)
{
  int home = homePe(id);
  if (home != CkMyPe())
  {
    thisProxy[home].requestLocation(id, CkMyPe());
  }
}

void CkLocCache::requestLocation(CmiUInt8 id, const int peToTell)
{
  if (peToTell == CkMyPe()) return;

  LocationMap::const_iterator itr = locMap.find(id);
  // TODO: If the location is not found, we probably need to buffer this request. Should
  // only effect very weird corner cases at the moment, and is a problem that already
  // existed, but should be addressed in the upcoming delivery/buffering cleanup.
  if (itr != locMap.end())
  {
    thisProxy[peToTell].updateLocation(itr->second);
  }
}

void CkLocCache::updateLocation(const CkLocEntry& newEntry)
{
  // printf("[%d] updateLocation: id=%llu pe=%d epoch=%d\n", CmiMyPe(), newEntry.id, newEntry.pe, newEntry.epoch);
  CkAssert(newEntry.pe != -1);
  // The answer is in; a later miss on this element may ask again.
  pendingLocReqs.erase(newEntry.id);
  CkLocEntry& oldEntry = locMap[newEntry.id];
  // printf("[%d] updateLocation: oldEntry.epoch=%d\n", CmiMyPe(), oldEntry.epoch);
  if (newEntry.epoch > oldEntry.epoch)
  {
    oldEntry = newEntry;
    notifyListeners(newEntry.id, newEntry.pe);
  }
}

void CkLocCache::recordEmigration(CmiUInt8 id, int pe)
{
  LocationMap::iterator itr = locMap.find(id);

  // printf("[%d] recordEmigration: id=%llu itr->second.pe=%d pe=%d\n", CmiMyPe(), id, itr->second.pe, pe);

  CkAssert(itr != locMap.end());
  CkAssert(itr->second.pe == CkMyPe());

  itr->second.pe = pe;
  itr->second.epoch++;
}


void CkLocCache::insert(CmiUInt8 id, int epoch)
{
  CkLocEntry& e = locMap[id];
  // TODO: This should be > probably, but demand creation needs some fixing up
  CkAssert(epoch >= e.epoch);
  e.id = id;
  e.pe = CkMyPe();
  e.epoch = epoch;
  notifyListeners(e.id, e.pe);
}

/*************************** LocMgr: CREATION *****************************/
CkLocMgr::CkLocMgr(CkArrayOptions opts)
    : bounds(opts.getBounds()),
      idCounter(1),
      thisProxy(thisgroup),
      thislocalproxy(thisgroup, CkMyPe())
{
  DEBC((AA "Creating new location manager %d\n" AB, thisgroup.idx));

  duringMigration = false;

  // Register with the map object
  mapID = opts.getMap();
  map = static_cast<CkArrayMap*>(CkLocalBranch(mapID));
  if (map == nullptr)
    CkAbort("ERROR! Local branch of array map is NULL!");
  // TODO: Should this be registered here, CkArray, CkLocMgr, or none?
  mapHandle = map->registerArray(opts.getEnd(), thisgroup);

  cacheID = opts.getLocationCache();
  cache = static_cast<CkLocCache*>(CkLocalBranch(cacheID));
  if (cache == nullptr)
    CkAbort("ERROR! Local branch of location cache is NULL!\n");

  // Figure out the mapping from indices to object IDs if one is possible
  compressor = ck::FixedArrayIndexCompressor::make(bounds);

  // Find and register with the load balancer
#if CMK_LBDB_ON
  lbmgrID = _lbmgr;
  metalbID = _metalb;
  initLB(lbmgrID, metalbID);
#endif
}

CkLocMgr::CkLocMgr(CkMigrateMessage* m)
    : IrrGroup(m), thisProxy(thisgroup), thislocalproxy(thisgroup, CkMyPe())
{
  duringMigration = false;
}

CkLocMgr::~CkLocMgr()
{
#if CMK_LBDB_ON
  syncBarrier->removeBeginReceiver(lbBarrierBeginReceiver);
  syncBarrier->removeEndReceiver(lbBarrierEndReceiver);
  lbmgr->UnregisterOM(myLBHandle);
#endif
  map->unregisterArray(mapHandle);
}

void CkLocMgr::pup(PUP::er& p)
{
  if (p.isPacking() && pendingImmigrate.size() > 0)
  {
    CkAbort(
        "Attempting to pup location manager with buffered migration messages."
        " Likely cause is checkpointing before array creation has fully completed\n");
  }
  IrrGroup::pup(p);
  p | mapID;
  p | mapHandle;
  p | cacheID;
  p | bounds;
  p | idCounter;
  if (p.isUnpacking())
  {
    thisProxy = thisgroup;
    CProxyElement_CkLocMgr newlocalproxy(thisgroup, CkMyPe());
    thislocalproxy = newlocalproxy;

    // Register with the map object
    map = static_cast<CkArrayMap*>(CkLocalBranch(mapID));
    if (map == nullptr)
      CkAbort("ERROR! Local branch of array map is NULL!");

    // Register with the cache object
    cache = static_cast<CkLocCache*>(CkLocalBranch(cacheID));
    if (cache == nullptr)
      CkAbort("ERROR! Local branch of location cache is NULL!");

    compressor = ck::FixedArrayIndexCompressor::make(bounds);
  }

#if CMK_LBDB_ON
  p | lbmgrID;
  p | metalbID;
  if (p.isUnpacking())
  {
    initLB(lbmgrID, metalbID);
  }
#endif

  // Delay doneInserting when it is unpacking during restart to prevent load balancing
  // from kicking in.
  if (p.isUnpacking() && !CkInRestarting())
  {
    doneInserting();
  }
}

/// Add a new local array manager to our list.
void CkLocMgr::addManager(CkArrayID id, CkArray* mgr)
{
  CK_MAGICNUMBER_CHECK
  DEBC((AA "Adding new array manager\n" AB));
  managers[id] = mgr;
  auto i = pendingImmigrate.begin();
  while (i != pendingImmigrate.end())
  {
    auto msg = *i;
    if (msg->nManagers <= managers.size())
    {
      i = pendingImmigrate.erase(i);
      immigrate(msg);
    }
    else
    {
      i++;
    }
  }
}

void CkLocMgr::deleteManager(CkArrayID id, CkArray* mgr)
{
  CkAssert(managers[id] == mgr);
  managers.erase(id);
  if (managers.empty())
    delete this;
}

// Tell this element's home processor it now lives "there"
void CkLocMgr::informHome(const CkArrayIndex& idx, int nowOnPe)
{
  int home = homePe(idx);
  // TODO: If home == CkMyPe() should we call update locally?
  if (home != CkMyPe() && home != nowOnPe)
  {
    // TODO: This may not need to be an idx update. Pretty sure the home will always
    // know the idx to id mapping.
    CmiUInt8 id = lookupID(idx);
    thisProxy[home].updateLocation(idx, cache->getLocationEntry(id));
  }
}

void CkLocMgr::noteImmigrationInFlight(CmiUInt8 id)
{
  if (inFlightImmigrations.insert(id).second && migDbg())
  {
    CkPrintf("[MIG %d] in-flight immigration id=%llu (%zu outstanding)\n", CkMyPe(),
             (unsigned long long)id, inFlightImmigrations.size());
    fflush(stdout);
  }
}

CkLocRec* CkLocMgr::createLocal(const CkArrayIndex& idx, bool forMigration,
                                bool ignoreArrival, bool notifyHome, int epoch)
{
  DEBC((AA "Adding new record for element %s\n" AB, idx2str(idx)));
  CmiUInt8 id = lookupID(idx);

  CkLocRec* rec = new CkLocRec(this, forMigration, ignoreArrival, idx, id);
  insertRec(rec, id);
  // The element is locally present from here on, so anything parked against it
  // can be released. This must precede updateLocation: that fires the location
  // listeners, and CkArray's listener replays bufferedIDMsgs through sendToPe,
  // which would park them right back if the id still read as in flight.
  if (inFlightImmigrations.erase(id) > 0 && migDbg())
  {
    CkPrintf("[MIG %d] immigration landed id=%llu (%zu outstanding)\n", CkMyPe(),
             (unsigned long long)id, inFlightImmigrations.size());
    fflush(stdout);
  }
  cache->insert(id, epoch);
  updateLocation(idx, cache->getLocationEntry(id));

  if (notifyHome)
  {
    informHome(idx, CkMyPe());
  }
  return rec;
}

// Used to handle messages that were buffered because of active rgets in progress
void CkLocMgr::processAfterActiveRgetsCompleted(CmiUInt8 id)
{
  // Call ckJustMigrated
  CkLocRec* myLocRec = elementNrec(id);
  callMethod(myLocRec, &CkMigratable::ckJustMigrated);
  callMethod(myLocRec, &CkMigratable::lbCheckWaitRelease);

  // Call ResumeFromSync on elements that were waiting for rgets
  auto iter2 = toBeResumeFromSynced.find(id);
  if (iter2 != toBeResumeFromSynced.end())
  {
    iter2->second->ResumeFromSync();
    toBeResumeFromSynced.erase(iter2);
  }

  // Deliver buffered messages to the elements that were waiting on rgets
  auto iter = bufferedActiveRgetMsgs.find(id);
  if (iter != bufferedActiveRgetMsgs.end())
  {
    std::vector<CkArrayMessage*> bufferedMsgs = iter->second;
    bufferedActiveRgetMsgs.erase(iter);
    for (auto msg : bufferedMsgs)
    {
      CmiHandleMessage(UsrToEnv(msg));
    }
  }
}

CmiUInt8 CkLocMgr::getNewObjectID(const CkArrayIndex& idx)
{
  CmiUInt8 id;
  if (!lookupID(idx, id))
  {
    id = idCounter++ + ((CmiUInt8)CkMyPe() << CMK_OBJID_ELEMENT_BITS);
    insertID(idx, id);
  }
  return id;
}

CkLocRec* CkLocMgr::registerNewElement(const CkArrayIndex& idx)
{
  CmiUInt8 id = getNewObjectID(idx);
  CkLocRec* rec = elementNrec(id);
  if (rec == nullptr)
  {
    // TODO: This is going to end up needlessly calling getNewObjectID(...) again
    rec = createLocal(idx, false, false, true);
  }

  return rec;
}

bool CkLocMgr::addElementToRec(CkLocRec* rec, CkArray* mgr, CkMigratable* elt,
                               int ctorIdx, void* ctorMsg)
{
  // Insert the new element into its manager's local list
  CmiUInt8 id = lookupID(rec->getIndex());
  if (mgr->getEltFromArrMgr(id))
  {
    CkAbort("Cannot insert array element twice!");
  }
  mgr->putEltInArrMgr(id, elt);  // Local element table

  // Call the element's constructor
  DEBC((AA "Constructing element %s of array\n" AB, idx2str(rec->getIndex())));
  CkMigratable_initInfo& i = CkpvAccess(mig_initInfo);
  i.locRec = rec;
  i.chareType = _entryTable[ctorIdx]->chareIdx;

#ifndef CMK_CHARE_USE_PTR
  int callingChareIdx = CkpvAccess(currentChareIdx);
  CkpvAccess(currentChareIdx) = -1;
#endif

  if (!rec->invokeEntry(elt, ctorMsg, ctorIdx, true))
    return false;

#ifndef CMK_CHARE_USE_PTR
  CkpvAccess(currentChareIdx) = callingChareIdx;
#endif

#if CMK_OUT_OF_CORE
  /* Register new element with out-of-core */
  PUP::sizer p_getSize;
  elt->virtual_pup(p_getSize);
  elt->prefetchObjID =
      CooRegisterObject(&CkArrayElementPrefetcher, p_getSize.size(), elt);
#endif

  return true;
}

// TODO: This might be not needed
void CkLocMgr::requestLocation(CmiUInt8 id)
{
  cache->requestLocation(id);
}

void CkLocMgr::requestLocation(const CkArrayIndex& idx)
{
  CmiUInt8 id;
  if (lookupID(idx, id))
  {
    requestLocation(id);
  }
  else
  {
    int home = homePe(idx);
    if (home != CkMyPe())
    {
      thisProxy[home].requestLocation(idx, CkMyPe());
    }
  }
}

bool CkLocMgr::requestLocation(const CkArrayIndex& idx, const int peToTell)
{
  CkAssert(peToTell != CkMyPe());

  CmiUInt8 id;
  if (lookupID(idx, id))
  {
    // We found the ID so update the location for peToTell
    thisProxy[peToTell].updateLocation(idx, cache->getLocationEntry(id));
    return true;
  }
  else
  {
    // We don't know the ID so buffer the location request
    DEBN(("%d Buffering ID/location req for %s\n", CkMyPe(), idx2str(idx)));
    bufferedLocationRequests[idx].push_back(peToTell);
    return false;
  }
}

void CkLocMgr::updateLocation(const CkArrayIndex& idx, const CkLocEntry& e)
{
  CkAssert(e.pe != -1);
  // Set the mapping from idx to id
  insertID(idx, e.id);

  // Update the location information
  cache->updateLocation(e);

  // Any location requests that we had to buffer because we didn't know how the index
  // mapped to the id can now be replied to.
  auto itr = bufferedLocationRequests.find(idx);
  if (itr != bufferedLocationRequests.end())
  {
    for (int pe : itr->second)
    {
      DEBN(("%d Replying to buffered ID/location req to pe %d\n", CkMyPe(), pe));
      if (pe != CkMyPe())
        thisProxy[pe].updateLocation(idx, e);
    }
    bufferedLocationRequests.erase(itr);
  }

  notifyListeners(idx, e.id, e.pe);
}

/*************************** LocMgr: DELETION *****************************/
// This index may no longer be used -- check if any of our managers are still
// using it, and if not delete it and clean up all traces of it on other PEs.
void CkLocMgr::reclaim(CkLocRec* rec)
{
  CK_MAGICNUMBER_CHECK
  // Return early if the record is still in use by any of our arrays
  for (auto itr = managers.begin(); itr != managers.end(); ++itr)
  {
    if (itr->second->lookup(rec->getID()))
      return;
  }
  removeFromTable(rec->getID());

  DEBC((AA "Destroying record for element %s\n" AB, idx2str(rec->getIndex())));
  if (!duringMigration)
  {  // This is a local element dying a natural death
    int home = homePe(rec->getIndex());
    if (home != CkMyPe())
#if CMK_MEM_CHECKPOINT
      if (!CkInRestarting())  // all array elements are removed anyway
#endif
        if (!duringDestruction)
          thisProxy[home].reclaimRemote(rec->getIndex(), CkMyPe());
  }
  delete rec;
}

// TODO: Rename to something more basic. This is just clearing entries from the
// cache. Will be called by reclaim remote, but maybe other things as well.
void CkLocMgr::reclaimRemote(const CkArrayIndex& idx, int deletedOnPe)
{
  DEBC((AA "Our element %s died on PE %d\n" AB, idx2str(idx), deletedOnPe));

  CmiUInt8 id;
  if (!lookupID(idx, id))
    CkAbort("Cannot find ID for the given index\n");

  // Delete the ID from location caching. Do not delete the idx to id mapping because
  // that remains constant and we don't want messages to end up stranded because they
  // can't figure out their index.
  cache->erase(id);
}

void CkLocMgr::removeFromTable(const CmiUInt8 id)
{
#if CMK_ERROR_CHECKING
  // Make sure it's actually in the table before we delete it
  if (NULL == elementNrec(id))
    CkAbort("CkLocMgr::removeFromTable called on invalid index!");
#endif
  hash.erase(id);
  clearResident(id);
  // Don't erase this during migration because the entry will be updated to reflect
  // the new location by calling recordEmigration
  if (!duringMigration)
    cache->erase(id);
#if CMK_ERROR_CHECKING
  // Make sure it's really gone
  if (NULL != elementNrec(id))
    CkAbort("CkLocMgr::removeFromTable called, but element still there!");
#endif
}

// This message took several hops to reach us-- fix it
void CkLocMgr::multiHop(CkArrayMessage* msg)
{
  CK_MAGICNUMBER_CHECK
  int srcPe = msg->array_getSrcPe();
  if (srcPe == CkMyPe())
    DEB((AA "Odd routing: local element %" PRIu64 " is %d hops away!\n" AB,
         msg->array_element_id(), msg->array_hops()));
  else
  {  // Send a routing message letting original sender know new element location
    DEBS((AA "Sending update back to %d for element %" PRIu64 "\n" AB, srcPe,
          msg->array_element_id()));
    cache->requestLocation(msg->array_element_id(), srcPe);
  }
}

bool CkLocMgr::checkInBounds(const CkArrayIndex& idx) const
{
  if (bounds.nInts > 0)
  {
    CkAssert(idx.dimension == bounds.dimension);
    bool shorts = idx.dimension > 3;

    for (int i = 0; i < idx.dimension; ++i)
    {
      unsigned int thisDim = shorts ? idx.indexShorts[i] : idx.index[i];
      unsigned int thatDim = shorts ? bounds.indexShorts[i] : bounds.index[i];
      if (thisDim >= thatDim) return false;
    }
  }
  return true;
}

/************************** LocMgr: ITERATOR *************************/
CkLocation::CkLocation(CkLocMgr* mgr_, CkLocRec* rec_) : mgr(mgr_), rec(rec_) {}

const CkArrayIndex& CkLocation::getIndex(void) const { return rec->getIndex(); }

CmiUInt8 CkLocation::getID() const { return rec->getID(); }

void CkLocation::destroyAll() { mgr->callMethod(rec, &CkMigratable::ckDestroy); }

void CkLocation::pup(PUP::er& p)
{
  mgr->pupElementsFor(p, rec, CkElementCreation_migrate);
}

CkLocIterator::~CkLocIterator() {}

/// Iterate over our local elements:
void CkLocMgr::iterate(CkLocIterator& dest)
{
  // Poke through the hash table for local ArrayRecs.
  for (LocRecHash::iterator it = hash.begin(); it != hash.end(); it++)
  {
    CkLocation loc(this, it->second);
    dest.addLocation(loc);
  }
}

/************************** LocMgr: MIGRATION *************************/
void CkLocMgr::pupElementsFor(PUP::er& p, CkLocRec* rec, CkElementCreation_t type,
                              bool rebuild)
{
  p.comment("-------- Array Location --------");

  // First pup the element types
  // (A separate loop so ckLocal works even in element pup routines)
  for (auto itr = managers.begin(); itr != managers.end(); ++itr)
  {
    int elCType;
    CkArray* arr = itr->second;
    if (!p.isUnpacking())
    {  // Need to find the element's existing type
      CkMigratable* elt = arr->getEltFromArrMgr(rec->getID());
      if (elt)
        elCType = elt->ckGetChareType();
      else
        elCType = -1;  // Element hasn't been created
    }
    p(elCType);
    if (p.isUnpacking() && elCType != -1)
    {
      // Create the element
      CkMigratable* elt = arr->allocateMigrated(elCType, type);
      int migCtorIdx = _chareTable[elCType]->getMigCtor();
      // Insert into our tables and call migration constructor
      if (!addElementToRec(rec, arr, elt, migCtorIdx, NULL))
        return;
      if (type == CkElementCreation_resume)
      {  // HACK: Re-stamp elements on checkpoint resume--
        //  this restores, e.g., reduction manager's gcount
        arr->stampListenerData(elt);
      }
    }
  }
  // Next pup the element data
  for (auto itr = managers.begin(); itr != managers.end(); ++itr)
  {
    CkMigratable* elt = itr->second->getEltFromArrMgr(rec->getID());
    if (elt != NULL)
    {
      elt->virtual_pup(p);
#if CMK_ERROR_CHECKING
      if (p.isUnpacking())
        elt->sanitycheck();
#endif
    }
  }
#if CMK_MEM_CHECKPOINT
  if (rebuild)
  {
    ArrayElement* elt;
    std::vector<CkMigratable*> list;
    migratableList(rec, list);
    CmiAssert(!list.empty());
    for (int l = 0; l < list.size(); l++)
    {
      //    reset, may not needed now
      // for now.
      for (int i = 0; i < CK_ARRAYLISTENER_MAXLEN; i++)
      {
        ArrayElement* elt = (ArrayElement*)list[l];
        contributorInfo* c = (contributorInfo*)&elt->listenerData[i];
        if (c)
          c->redNo = 0;
      }
    }
  }
#endif
}

/// Call this member function on each element of this location:
void CkLocMgr::callMethod(CkLocRec* rec, CkMigratable_voidfn_t fn)
{
  for (auto itr = managers.begin(); itr != managers.end(); ++itr)
  {
    CkMigratable* el = itr->second->getEltFromArrMgr(rec->getID());
    if (el)
      (el->*fn)();
  }
}

/// Call this member function on each element of this location:
void CkLocMgr::callMethod(CkLocRec* rec, CkMigratable_voidfn_arg_t fn, void* data)
{
  for (auto itr = managers.begin(); itr != managers.end(); ++itr)
  {
    CkMigratable* el = itr->second->getEltFromArrMgr(rec->getID());
    if (el)
      (el->*fn)(data);
  }
}

/// return a list of migratables in this local record
void CkLocMgr::migratableList(CkLocRec* rec, std::vector<CkMigratable*>& list)
{
  for (auto itr = managers.begin(); itr != managers.end(); ++itr)
  {
    CkMigratable* elt = itr->second->getEltFromArrMgr(rec->getID());
    if (elt)
      list.push_back(elt);
  }
}

bool did_inter_node_gpudirect_rdma(int srcPe, int dstPe) {
  CmiEnforce((srcPe >= 0) && (srcPe <= CmiNumPes()));
  CmiEnforce((dstPe >= 0) && (dstPe <= CmiNumPes()));

  if (CmiNodeOf(srcPe) == CmiNodeOf(dstPe)) {
    return false;
  } else if (CmiPeOnSamePhysicalNode(srcPe, dstPe)) {
    return false;
  } else {
    return true;
  }
}

#if CMK_CUDA
void CkLocMgr::sendGPUMsg(CmiUInt8 id)
{
  if (migDbg())
    CkPrintf("[GPUSEND %d] dispatch id=%llu\n", CkMyPe(), (unsigned long long)id);
  auto gpuData = sendGPUBuffers[id];

  // No completion callback and no blocking wait. The previous
  // CkCallbackResumeThread parked this (threaded) entry's thread until the
  // whole remote transfer completed, once per migrating chare -- serializing
  // cross-node migrations behind their own transfers. The source pool block's
  // lifetime is instead handled per transport: a same-node staged send
  // registers the block with its IPC event and reclaimCompletedIpcEvents
  // frees it once the receiver has read it; an inter-node send is acked
  // explicitly -- the receiver's immigrateGPU entry runs only after the rget
  // has landed, and its finishGPUSend(id) releases the block here.
  // This send belongs to the RUNTIME, not to whatever element last ran on
  // this PE: _currentLocRec can still name one here, and attributing the
  // payload send to it corrupts that element's migration interlock -- and,
  // with the registration cache on, registers the LB-buffer block against an
  // owner whose own migration then deregisters it mid-transfer, wedging this
  // immigration permanently ("in-flight immigration" that never lands).
  {
    CkLocRec* saved_rec = CkpvAccess(_currentLocRec);
    if (saved_rec != NULL && getenv("CHARM_DEBUG_MIGRATE"))
      CkPrintf("[GPUSEND %d] stale _currentLocRec at payload send (id=%llu)\n",
               CkMyPe(), (unsigned long long)id);
    CkpvAccess(_currentLocRec) = NULL;
    thisProxy[gpuData.toPe].immigrateGPU(id, gpuData.size,
      CkDeviceBuffer(gpuData.data, gpuData.size), CkMyPe());
    CkpvAccess(_currentLocRec) = saved_rec;
  }

  // Who releases the staged block depends on how the send actually travelled.
  //
  // Same node, different process: the send goes through CUDA IPC, which
  // registers the block against an IPC event, and reclaimCompletedIpcEvents
  // frees it once the receiver sets dst_flag. Hand it over.
  //
  // Same process: the send is a plain device-to-device memcpy. Nothing
  // registers the block with an event, so the reclaim scan can never see it --
  // measured, 213 allocations produced zero reclaims, which exhausted the load
  // balance buffer after two steps at 128MB. Keep the entry and let the
  // receiver's ack free it, as inter-node already does.
  const bool same_process = (CmiNodeOf(CkMyPe()) == CmiNodeOf(gpuData.toPe));
  if (!did_inter_node_gpudirect_rdma(CkMyPe(), gpuData.toPe) && !same_process) {
    sendGPUBuffers.erase(id);
  }
  // Otherwise the entry stays in sendGPUBuffers until finishGPUSend(id).
}

// Inter-node ack: the receiver's payload has fully arrived, so the packed
// source block this migration was holding can be released.
void CkLocMgr::finishGPUSend(CmiUInt8 id)
{
  auto it = sendGPUBuffers.find(id);
  if (it == sendGPUBuffers.end()) return;

  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  if (csv_gpu_manager.use_shm) {
    DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];
#if CMK_SMP
    CmiLock(dm->lock);
#endif
    dm->free_comm_buffer((size_t)((char*)it->second.data - (char*)dm->comm_buffer->base_ptr));
#if CMK_SMP
    CmiUnlock(dm->lock);
#endif
  }
  sendGPUBuffers.erase(it);
}
#endif

/// Migrate this local element away to another processor.
// Fast path for intra-process (same CmiNode) migration in SMP mode.
// Transfers ownership of the live CkMigratable objects to the destination PE
// without pup-based serialization. Must be called after ckAboutToMigrate and
// only when there is no GPU-resident data.
bool CkLocMgr::emigrateIntraProcess(CkLocRec* rec, int toPe)
{
  CkArrayIndex idx = rec->getIndex();
  CmiUInt8 id = rec->getID();

  // Gather (aid, elt) pairs from each manager and detach each chare from
  // this PE's sync barrier, capturing the barrier epoch.
  std::vector<CkGroupID> aids;
  std::vector<uintptr_t> eltPtrs;
  aids.reserve(managers.size());
  eltPtrs.reserve(managers.size());
  int barrierEpoch = -1;

  for (auto itr = managers.begin(); itr != managers.end(); ++itr)
  {
    CkMigratable* elt = itr->second->getEltFromArrMgr(id);
    if (!elt) continue;
    int ep = elt->ckPrepareIntraProcessMigrate();
    if (ep != -1) barrierEpoch = ep;
    aids.push_back(itr->first);
    eltPtrs.push_back(reinterpret_cast<uintptr_t>(elt));
  }

  int nElts = (int)aids.size();
  int locEpoch = cache->getEpoch(id) + 1;

  CkArrayElementIntraProcessMigrateMessage* msg =
      new (nElts, nElts) CkArrayElementIntraProcessMigrateMessage(
          idx, id,
#if CMK_LBDB_ON
          rec->isAsyncMigrate(),
#else
          false,
#endif
          nElts, locEpoch, barrierEpoch);
  for (int i = 0; i < nElts; i++)
  {
    msg->aids[i] = aids[i];
    msg->eltPtrs[i] = eltPtrs[i];
  }

  DEBM((AA "Intra-process migrating index %s to %d\n" AB, idx2str(idx), toPe));

  thisProxy[toPe].immigrateIntraProcess(msg);

  // Detach from this PE's array managers and location tables. Do NOT delete
  // the CkMigratable objects -- they now live on the destination PE.
  duringMigration = true;
  for (auto itr = managers.begin(); itr != managers.end(); ++itr)
  {
    itr->second->eraseEltFromArrMgr(id);
  }
  removeFromTable(id);
  delete rec;
  duringMigration = false;

  cache->recordEmigration(id, toPe);
  informHome(idx, toPe);
  // Anything admission control held for this element can now go back through
  // delivery, which will forward it to the new home.
  CkDeviceRecvAdmissionReplay(id);

#if !CMK_LBDB_ON && CMK_GLOBAL_LOCATION_UPDATE
  CmiPrintf((AA "Global location update. idx %s "
           "assigned to %d \n" AB,
        idx2str(idx), toPe));
  thisProxy.updateLocation(id, toPe);
#endif

  CK_MAGICNUMBER_CHECK
  return true;
}

// Receive side of the intra-process migration fast path.
void CkLocMgr::immigrateIntraProcess(CkArrayElementIntraProcessMigrateMessage* msg)
{
  MigTimer _t(&g_mig.intraSecs, &g_mig.intraN);
  const CkArrayIndex& idx = msg->idx;

  if (msg->nManagers > (int)managers.size())
  {
    // Some array managers haven't registered yet. This should be rare during
    // an AtSync-driven migration; fall back to the existing pendingImmigrate
    // list is not yet implemented, so abort loudly to catch it in testing.
    CkAbort("Intra-process immigrate: managers not yet registered on destination PE\n");
  }

  insertID(idx, msg->id);

  CkLocRec* rec = elementNrec(msg->id);
  if (rec == nullptr)
    rec = createLocal(idx, true /* fromMigration */, msg->ignoreArrival,
                      false /* home told on departure */, msg->locEpoch);

  for (int i = 0; i < msg->nManagers; i++)
  {
    auto mit = managers.find(msg->aids[i]);
    if (mit == managers.end())
      CkAbort("Intra-process immigrate: unknown array manager\n");
    CkMigratable* elt = reinterpret_cast<CkMigratable*>(msg->eltPtrs[i]);
    mit->second->putEltInArrMgr(msg->id, elt);
    elt->ckFinalizeIntraProcessMigrate(rec, msg->barrierEpoch);
  }

  callMethod(rec, &CkMigratable::ckJustMigrated);
  callMethod(rec, &CkMigratable::lbCheckWaitRelease);

  delete msg;
}

void CkLocMgr::emigrate(CkLocRec* rec, int toPe)
{
  CK_MAGICNUMBER_CHECK
  if (toPe == CkMyPe())
    return;  // You're already there!

  // Do not move an element while the runtime is still reading buffers it handed
  // to a zerocopy send: migration frees and reallocates exactly those buffers.
  // Stand the migration down and let noteDeviceSendDone re-drive it when the
  // last transfer completes.
  // Under the split barrier, only a parked element moves. A running element
  // frozen in place mid-step deadlocks in cycles -- it stalls on receives
  // from neighbours that are themselves frozen awaiting their own moves, which
  // a batch decision (every element at once) makes certain. The park is the
  // one point where the element provably holds no unconsumed device receive
  // (CkDeviceRecvQuiet gates it), runs nothing, and admission control bounces
  // everything new -- quiesced by construction, exactly what a move needs.
  if (rec->outstandingDeviceSends > 0
#if CMK_LBDB_ON
      || (_lb_args.lbAsync() && !rec->deviceRecvParked)
#endif
     )
  {
    if (getenv("CHARM_DEBUG_MIGRATE"))
      CmiPrintf("[DEFER-MIG %d] elem %s count=%d parked=%d toPe=%d\n", CkMyPe(),
                idx2str(rec->getIndex()), rec->outstandingDeviceSends,
                (int)rec->deviceRecvParked, toPe);
    rec->pendingMigrateTo = toPe;
    return;
  }

#if (CMK_CUDA || CMK_HIP) && CMK_GPU_COMM
  // Those same buffers are about to be freed, so any registration cached for
  // them dies with them. Safe here and only here: the check above is what
  // guarantees nothing is reading them.
  CkRdmaDeviceDropRegistrations(rec);
#endif

  CkArrayIndex idx = rec->getIndex();
  CmiUInt8 id = rec->getID();

#if CMK_OUT_OF_CORE
  /* Load in any elements that are out-of-core */
  for (auto itr = managers.begin(); itr != managers.end(); ++itr)
  {
    CkMigratable* el = itr->second->getEltFromArrMgr(rec->getID());
    if (el)
      if (!el->isInCore)
        CooBringIn(el->prefetchObjID);
  }
#endif

  // Let all the elements know we're leaving
  callMethod(rec, &CkMigratable::ckAboutToMigrate);

  // First pass: find size of migration message
  size_t bufSize, gpuBufSize;
  PUP::sizer p(PUP::er::IS_MIGRATION);
  pupElementsFor(p, rec, CkElementCreation_migrate);
  bufSize = p.size();

  gpuBufSize = 0;
  size_t nGpuBufs = 0;
#if CMK_CUDA
  gpuBufSize = p.gpu_size();
  nGpuBufs = p.gpu_buf_count();
#endif

  // Fast path: for intra-process (same CmiNode) migration, transfer ownership
  // of the live chare objects to the destination PE without packing/copying.
  // GPU-resident data is only carried along when the destination PE is mapped
  // to the same physical GPU as the source, because raw device pointers are
  // only directly usable in that case.
  // Escape hatch for bisecting migration problems: setting
  // CHARM_NO_INTRAPROC_MIGRATE forces the packed path for every migration.
  static const bool intraProcDisabled =
      (getenv("CHARM_NO_INTRAPROC_MIGRATE") != nullptr);
  if (!intraProcDisabled && CmiNodeOf(CkMyPe()) == CmiNodeOf(toPe))
  {
    bool sameGpuRequirementMet = (gpuBufSize == 0);
#if CMK_CUDA
    if (gpuBufSize > 0)
      sameGpuRequirementMet = (hapiDeviceForPe(CkMyPe()) == hapiDeviceForPe(toPe));
#endif
    if (sameGpuRequirementMet && emigrateIntraProcess(rec, toPe))
      return;
  }

  // Anything the intra-process handoff above did not take stages from here --
  // including a move within this process onto a different GPU. See the note on
  // migration transports below for why, and what supporting that case directly
  // would require.

#if CMK_ERROR_CHECKING
  if (bufSize > std::numeric_limits<int>::max())
  {
    CkAbort("Cannot migrate an object with size greater than %d bytes!\n",
            std::numeric_limits<int>::max());
  }
#endif


  void* gpuMsg = nullptr;
#if CMK_CUDA
  hapiStream_t migStream = NULL;
#endif
  // Allocate and pack into message
  CkArrayElementMigrateMessage* msg =
      new (bufSize, nGpuBufs, 0) CkArrayElementMigrateMessage(idx, id,
#if CMK_LBDB_ON
                                                    rec->isAsyncMigrate(),
#else
                                                    false,
#endif
                                                    bufSize, managers.size(),
                                                    cache->getEpoch(id) + 1,
                                                    gpuBufSize > 0,
                                                    (int)nGpuBufs);

  {
#if CMK_CUDA
    if (gpuBufSize > 0) {
      GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
      if(csv_gpu_manager.use_shm) {
        DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];
        // Takes dm->lock itself, and reclaims retired IPC slots before giving
        // up -- this region is only ever freed by that scan.
        gpuMsg = CkRdmaDeviceAllocLbBuffer(dm, gpuBufSize);
        if (gpuMsg == nullptr) {
          CkAbort("PE %d, device %d: Not enough memory on device Load balance buffer (%zu free)",
              CkMyPe(), dm->global_index, dm->get_lb_buffer_free_size());
        }
#if CMK_SMP
        CmiLock(dm->lock);
#endif
        // Pack copies go on the device's migration stream, not the null
        // stream (see DeviceManager::migration_stream). Lazy creation is
        // serialized by this lock.
        if (dm->migration_stream == NULL)
          hapiCheck(hapiStreamCreate(&dm->migration_stream));
        migStream = dm->migration_stream;
#if CMK_SMP
        CmiUnlock(dm->lock);
#endif
      }
    }
#endif
    PUP::toMem p(msg->packData, gpuMsg, PUP::er::IS_MIGRATION);
#if CMK_CUDA
    p.gpuStream = (void*)migStream;
    p.deviceManifestOut = msg->gpuManifest;
    p.deviceManifestCap = nGpuBufs;
#endif
    p.becomeDeleting();
    pupElementsFor(p, rec, CkElementCreation_migrate);
    if (p.size() != bufSize)
    {
      CkError(
          "ERROR! Array element claimed it was %zu bytes to a "
          "sizing PUP::er, but copied %zu bytes into the packing PUP::er!\n",
          bufSize, p.size());
      CkAbort("Array element's pup routine has a direction mismatch.\n");
    }
#if CMK_CUDA
    // The same check for the device stream. Both totals matter and neither is
    // implied by the host size: gpu_size() sizes the device region the pack
    // copies into, and gpu_buf_count() sizes msg->gpuManifest, a host array
    // inside this message. Under-counting either one is a buffer overflow --
    // of device memory, or of the heap -- that shows up much later as
    // unrelated corruption, so catch it where the mismatch actually is.
    // Overflow only: a pass that packs *less* than it sized wastes room but
    // corrupts nothing, and the sizer may legitimately round up trailing
    // alignment. Packing more is the bug worth aborting on.
    if (p.gpu_size() > gpuBufSize || p.deviceManifestIdx > nGpuBufs)
    {
      CkError(
          "ERROR! Array element claimed %zu device bytes in %zu buffers to a "
          "sizing PUP::er, but packed %zu bytes in %zu buffers!\n",
          gpuBufSize, nGpuBufs, p.gpu_size(), p.deviceManifestIdx);
      CkAbort("Array element's pup routine has a device direction mismatch.\n");
    }
#endif
  }

  DEBM((AA "Migrated index size %s to %d \n" AB, idx2str(idx), toPe));

#if CMK_CUDA
  // Complete the pack copies before destroying elements. On the migration
  // stream this waits for exactly those copies instead of the whole device;
  // the device-wide fallback covers the no-pool path, whose packing still
  // used the null stream (and cudaMemcpy(D2D) is async in CUDA 12.x).
  if (gpuBufSize > 0) {
    if (migStream != NULL)
      hapiCheck(hapiStreamSynchronize(migStream));
    else
      cudaDeviceSynchronize();
  }
#endif

  thisProxy[toPe].immigrate(msg);

  duringMigration = true;
  for (auto itr = managers.begin(); itr != managers.end(); ++itr)
  {
    itr->second->deleteElt(id);
  }
  duringMigration = false;

  cache->recordEmigration(id, toPe);
  informHome(idx, toPe);
#if CMK_CUDA
  if (gpuBufSize > 0)
  {
    sendGPUBuffers[id] = GPUMigrateData(toPe, gpuBufSize, gpuMsg);
    thisProxy[CkMyPe()].sendGPUMsg(id);
  }
  if (getenv("CHARM_DEBUG_MIGRATE"))
    CmiPrintf("[EMIG %d] id=%llu gpuBufSize=%zu toPe=%d\n", CkMyPe(),
              (unsigned long long)id, (size_t)gpuBufSize, toPe);
#endif
  // Anything admission control held for this element goes back through
  // delivery, which forwards it to the new home.
  //
  // This runs AFTER the GPU payload is dispatched, and the order is load
  // bearing. Both the replay and sendGPUMsg go onto this PE's own queue,
  // which the scheduler drains ahead of the network. Replaying first puts
  // every message the element buffered while parked in front of the send
  // that ships its device buffers, so the destination waits forever for a
  // payload that is stuck behind traffic addressed to the element that
  // already left -- the migration never completes, its PE never reports in,
  // and the balancer's resume reduction hangs with it.
  CkDeviceRecvAdmissionReplay(id);

#if !CMK_LBDB_ON && CMK_GLOBAL_LOCATION_UPDATE
  CmiPrintf((AA "Global location update. idx %s "
           "assigned to %d \n" AB,
        idx2str(idx), toPe));
  thisProxy.updateLocation(id, toPe);
#endif

  CK_MAGICNUMBER_CHECK
}

#if CMK_LBDB_ON
void CkLocMgr::informLBPeriod(CkLocRec* rec, int lb_ideal_period)
{
  callMethod(rec, &CkMigratable::recvLBPeriod, (void*)&lb_ideal_period);
}

void CkLocMgr::metaLBCallLB(CkLocRec* rec)
{
  callMethod(rec, &CkMigratable::metaLBCallLB);
}
#endif

// Device migration transports: staged, plus a same-process pointer handoff.
//
// emigrateIntraProcess hands the live chare over when source and destination
// PEs are in one process AND on one GPU. Everything else stages: the chare's
// device state is packed into a pool block, the chare is destroyed at pack, and
// the block travels as an ordinary device zerocopy payload.
//
// Same process but a different GPU therefore stages too, at the cost of one
// device-to-device copy. Supporting it directly -- the destination reading the
// source's buffers where they lie -- was tried and removed, because it is not
// merely a pointer handoff: a cross-GPU read is asynchronous, so the source
// chare has to outlive it. Reinstating it needs all of:
//
//   - a pack that records the chare's device buffers as (ptr,size) instead of
//     copying them (a collector hook on PUP::er),
//   - a message carrying those handles to the destination,
//   - a deferred-destruction table on the source, since the element must not be
//     deleted at emigrate,
//   - an ack from the destination once its copies retire, to drive that table,
//   - an explicit erase from CkpvAccess(array_objs) at emigrate: ~ArrayElement
//     normally does it, and an undestroyed element would otherwise keep
//     receiving messages through the PE-level fast path, and
//   - accounting in the LB memory contract, which must treat the move as a held
//     transport debiting both ends -- see heldTransport(), whose held/staged
//     line has to keep matching whatever this file routes.
//
// The reason it is not worth it by default is the last two: holding device
// memory until a peer acks is unbounded in the number of concurrent migrations,
// while staging frees the source at pack and keeps a step's footprint inside a
// reserve the balancer can plan against.

#if CMK_CUDA
// Arena mode (CHARM_MIGRATE_ARENA): the staged payload lands in one ordinary
// device allocation that becomes the arriving chare's permanent storage --
// pup_buffer_device rebinds pointers into it at unpack, and nothing is
// copied. The arena needs only the payload's size, which this message
// carries, so it works with either arrival order of host message and
// payload. Off by default while experimental.
static inline bool ckMigrateArenaMode()
{
  static const bool on = (getenv("CHARM_MIGRATE_ARENA") != nullptr);
  return on;
}

void CkLocMgr::immigrateGPU(CmiUInt8& id, int& size, char* &data, int& srcPe, CkDeviceBufferPost* post)
{
  //CkPrintf("PE %d allocating GPU memory size %d for id %llu\n", CkMyPe(), size, id);
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  if (ckMigrateArenaMode()) {
    // The arena: the chare's storage, not a landing block, so it comes from
    // ordinary device memory -- pool memory is transient by contract.
    data = nullptr;
    hapiCheck(hapiMalloc((void**)&data, size));
  } else if(csv_gpu_manager.use_shm) {
    DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];
    data = (char*)CkRdmaDeviceAllocLbBuffer(dm, size);
    if (data == nullptr) {
      CkAbort("PE %d, device %d: Not enough memory on device Load balance buffer (%zu free)",
          CkMyPe(), dm->global_index, dm->get_lb_buffer_free_size());
    }
  }
  // No barrier here. Every path that returns a block to this pool already hands
  // back memory with no device work in flight against it: the IPC reclaim
  // queries the transfer's event, finishGPUSend waits for the receiver's ack, a
  // block claimed without an event slot is handed back untouched, and the
  // landing block freed after unpack is covered by the device sync in
  // immigrate. Arena mode allocates fresh memory and cannot alias at all.
  //
  // That last one is why this is worth a note: the sync in immigrate and the
  // barrier that used to stand here were redundant with each other, and with
  // this one gone that sync is load-bearing. Narrowing it to a stream sync
  // means giving the free after unpack its own gate at the same time.
  // The receive lands on this PE's per-thread stream, which is what every other
  // device operation in the runtime uses, and the unpack copies ride the same
  // stream (see immigrate) -- so the read out of this block is ordered behind
  // the write into it by the stream, not by a barrier.
  //
  // It used to post on cudaStream_t(0), the legacy default stream, since
  // nothing here builds with --default-stream per-thread. A per-thread stream
  // is by definition non-blocking against the legacy one, so that left the
  // landing write unordered against every other PE's work in the process, and a
  // context-wide cudaDeviceSynchronize() stood here to cover the gap. It was
  // load-bearing: removing it while still posting on stream 0 corrupted 4 of 11
  // runs at two PEs per process (twice the same wrong checksum), and never
  // reproduced at one PE per process -- which is the clue that it was about two
  // PEs sharing a context, not about this pool's free paths, all of which are
  // already gated.
  // This barrier's necessity is unresolved. Removing it correlated with wrong
  // checksums at two PEs per process (4 of 11 runs) in one afternoon's testing;
  // a later reconstruction of that same build could not reproduce it at all
  // across ~70 runs, and a three-step bisect cleared every intervening change.
  // So the evidence for it is a sample that never reproduced again, and the
  // measurements it rests on could not distinguish the rates involved.
  //
  // It stays because nothing has shown it to be unnecessary, and because the
  // failure it may guard is silent data corruption. Anyone removing it needs a
  // reproducer that reproduces -- which nobody currently has -- not another
  // eight-run sample. Ruled out as explanations, each by measurement: comm
  // buffer reuse under live work, the pool's four free paths, per-thread vs
  // legacy stream ordering here, the shared migration stream, IPC slot
  // exhaustion. Note also that any added synchronization near this window hides
  // the failure, so probes placed inside it are worthless.
  cudaDeviceSynchronize();
  receivedDeviceMsgs[id] = data;
  post[0].hapi_stream = (cudaStream_t) 0;
}



void CkLocMgr::immigrateGPU(CmiUInt8 id, int size, char* data, int srcPe)
{
  if (migDbg())
    CkPrintf("[GPURECV %d] id=%llu size=%d srcPe=%d\n", CkMyPe(),
             (unsigned long long)id, size, srcPe);
  // This entry runs only after the device payload has fully landed, so an
  // inter-node sender's packed source block is safe to release now. Same-node
  // sends release theirs through the IPC event reclaim instead.
  // Ack for the two cases where the sender still holds the staged block:
  // inter-node, and same-process, whose memcpy send registers no IPC event so
  // nothing else would ever free it.
  if (srcPe >= 0 && (did_inter_node_gpudirect_rdma(srcPe, CkMyPe()) ||
                     CmiNodeOf(srcPe) == CmiNodeOf(CkMyPe())))
    thisProxy[srcPe].finishGPUSend(id);

  void* dataPtr = receivedDeviceMsgs[id];
  receivedDeviceMsgs.erase(id);
  bufferedDeviceMigrateMsgs[id] = dataPtr;
  if (bufferedHostMigrateMsgs.find(id) != bufferedHostMigrateMsgs.end())
  {
    immigrate(bufferedHostMigrateMsgs[id]);
    bufferedHostMigrateMsgs.erase(id);
  }
}
#endif

/**
  Migrating array element is arriving on this processor.
*/
void CkLocMgr::immigrate(CkArrayElementMigrateMessage* msg)
{
  MigTimer _t(&g_mig.packedSecs, &g_mig.packedN);
  void* gpuMsg = nullptr;
  if (msg->hasGPUMsg)
  {
    auto it = bufferedDeviceMigrateMsgs.find(msg->id);

    if (it == bufferedDeviceMigrateMsgs.end())
    {
      // Host state is here but the device pull is not. Park the message and
      // record that the element is mid-flight into this PE.
      bufferedHostMigrateMsgs[msg->id] = msg;
      noteImmigrationInFlight(msg->id);
      return;
    }
    gpuMsg = it->second;
  }

  const CkArrayIndex& idx = msg->idx;

  PUP::fromMem p(msg->packData, gpuMsg, PUP::er::IS_MIGRATION);
#if CMK_CUDA
  // Sender-recorded device manifest: every DEVICE-mode unpack call is checked
  // against it (see PUP::fromMem::deviceManifestIn). nGpuBufs is 0 for
  // migrations whose payload was not packed by toMem (the by-handle path).
  if (msg->hasGPUMsg && msg->nGpuBufs > 0) {
    p.deviceManifestIn = msg->gpuManifest;
    p.deviceManifestCount = (size_t)msg->nGpuBufs;
  }
  // Arena mode applies only to staged payloads (nGpuBufs > 0 identifies
  // them; a by-handle payload landed in a pool block regardless of the env
  // flag). pup_buffer_device calls then rebind into gpuMsg instead of
  // copying out of it.
  const bool gpuMsgIsArena =
      ckMigrateArenaMode() && msg->hasGPUMsg && msg->nGpuBufs > 0 &&
      gpuMsg != nullptr;
  if (gpuMsgIsArena) p.deviceRebind = true;
#endif

  if (msg->nManagers < managers.size())
    CkAbort("Array element arrived from location with fewer managers!\n");
  if (msg->nManagers > managers.size())
  {
    // Some array managers haven't registered yet -- buffer the message
    DEBM((AA "Buffering %s immigrate msg waiting for array registration\n" AB,
          idx2str(idx)));
    pendingImmigrate.push_back(msg);
    noteImmigrationInFlight(msg->id);
    return;
  }

  if (msg->hasGPUMsg)
    bufferedDeviceMigrateMsgs.erase(msg->id);

  insertID(idx, msg->id);

  // Create a record for this element
  CkLocRec* rec = elementNrec(msg->id);
  if (rec == nullptr)
      rec = createLocal(idx, true, msg->ignoreArrival, false /* home told on departure */, msg->epoch);

  CmiAssert(CpvAccess(newZCPupGets).empty());  // Ensure that vector is empty
  // Create the new elements as we unpack the message
  pupElementsFor(p, rec, CkElementCreation_migrate);
#if CMK_CUDA
  // Those copies may still be in flight and the element must not run until its
  // device state is complete, but wait on the one stream that carried them
  // rather than the whole device: as a device-wide barrier this ran once per
  // arriving object and was the bulk of the cost of the first step after load
  // balancing. It also gates the free of gpuMsg below.
  if (gpuMsg != nullptr)
    cudaDeviceSynchronize();
#endif

#if CMK_CUDA
  GPUManager& csv_gpu_manager = CsvAccess(gpu_manager);
  // Only elements that actually carried device state allocated a comm-buffer
  // block above (gpuMsg stays null when msg->hasGPUMsg is false). Freeing
  // unconditionally handed the allocator base_ptr + (0 - base_ptr) == nullptr
  // and aborted, so migrating any element without GPU state was fatal.
  if (gpuMsgIsArena) {
    if (p.deviceReboundCount > 0) {
      // The chare's rebound buffers live in the arena: keep it, and let
      // hapiFreeMigratable release it when the last one is freed. Extent is
      // the packed stream's aligned extent, recomputed from the manifest.
      size_t extent = 0;
      for (int i = 0; i < msg->nGpuBufs; ++i)
        extent = PUP::alignDeviceOffset(extent) + msg->gpuManifest[i];
      hapiArenaRegister(gpuMsg, extent, p.deviceReboundCount);
    } else {
      // Nothing rebound: the arena served as a plain landing area for a
      // legacy pup, and its contents were copied out.
      hapiCheck(hapiFree(gpuMsg));
    }
  } else if(csv_gpu_manager.use_shm && gpuMsg != nullptr) {
    DeviceManager* dm = csv_gpu_manager.device_map[CkMyPe()];
#if CMK_SMP
    CmiLock(dm->lock);
#endif
  dm->free_comm_buffer((size_t)((char*)gpuMsg - (char*)dm->comm_buffer->base_ptr));
#if CMK_SMP
    CmiUnlock(dm->lock);
#endif
  }
#endif

  bool zcRgetsActive = !CpvAccess(newZCPupGets).empty();
  if (zcRgetsActive)
  {
    // newZCPupGets is not empty, rgets need to be launched
    // newZCPupGets is populated with NcpyOperationInfo during pupElementsFor by
    // pup_buffer calls that require Rgets Issue Rgets using the populated newZCPupGets
    // vector
    zcPupIssueRgets(msg->id, this);
  }
  CpvAccess(newZCPupGets).clear();  // Clear this to reuse the vector
#if CMK_CUDA
  // The element-wise check catches wrong sizes; this catches an unpack that
  // asked for fewer device buffers than were packed -- also a direction
  // mismatch, just one the per-call check cannot see.
  if (p.deviceManifestCount > 0 && p.deviceManifestIdx != p.deviceManifestCount)
    CkAbort("Device pup direction mismatch: sender packed %zu device buffers "
            "but unpack consumed only %zu",
            p.deviceManifestCount, p.deviceManifestIdx);
#endif
  if (p.size() != msg->length)
  {
    CkError(
        "ERROR! Array element claimed it was %d bytes to a"
        "packing PUP::er, but %zu bytes in the unpacking PUP::er!\n",
        msg->length, p.size());
    CkError("(I have %zu managers; it claims %d managers)\n", managers.size(),
            msg->nManagers);

    CkAbort("Array element's pup routine has a direction mismatch.\n");
  }

  if (migDbg())
    CkPrintf("[JUSTMIG %d] elem %s zcRgetsActive=%d\n", CkMyPe(),
             idx2str(idx), (int)zcRgetsActive);
  if (!zcRgetsActive)
  {
    // Let all the elements know we've arrived
    callMethod(rec, &CkMigratable::ckJustMigrated);
    callMethod(rec, &CkMigratable::lbCheckWaitRelease);
  }

  delete msg;
}

void CkLocMgr::restore(const CkArrayIndex& idx, CmiUInt8 id, PUP::er& p)
{
  insertID(idx, id);

  CkLocRec* rec = createLocal(idx, false, false, false);

  // Create the new elements as we unpack the message
  pupElementsFor(p, rec, CkElementCreation_restore);

  callMethod(rec, &CkMigratable::ckJustRestored);
}

/// Insert and unpack this array element from this checkpoint (e.g., from CkLocation::pup)
void CkLocMgr::resume(const CkArrayIndex& idx, CmiUInt8 id, PUP::er& p, bool notify,
                      bool rebuild)
{
  insertID(idx, id);

  CkLocRec* rec = createLocal(idx, false, false, notify /* home doesn't know yet */);

  // Create the new elements as we unpack the message
  pupElementsFor(p, rec, CkElementCreation_resume, rebuild);

  callMethod(rec, &CkMigratable::ckJustMigrated);
}

/********************* LocMgr: UTILITY ****************/
void CkMagicNumber_impl::badMagicNumber(int expected, const char* file, int line,
                                        void* obj) const
{
  CkError(
      "FAILURE on pe %d, %s:%d> Expected %p's magic number "
      "to be 0x%08x; but found 0x%08x!\n",
      CkMyPe(), file, line, obj, expected, magic);
  CkAbort(
      "Bad magic number detected!  This implies either\n"
      "the heap or a message was corrupted!\n");
}
CkMagicNumber_impl::CkMagicNumber_impl(int m) : magic(m) {}

/// Return true if this array element lives on another processor
bool CkLocMgr::isRemote(const CkArrayIndex& idx, int* onPe) const
{
  int pe = whichPe(idx);
  /* not definitely a remote element */
  if (pe == -1 || pe == CkMyPe())
    return false;
  // element is indeed remote
  *onPe = pe;
  return true;
}

static const char* rec2str[] = {
    "base (INVALID)",  // Base class (invalid type)
    "local",           // Array element that lives on this Pe
};

// If we are deleting our last array manager set duringDestruction to true to
// avoid sending out unneeded reclaimRemote messages.
void CkLocMgr::setDuringDestruction(bool _duringDestruction)
{
  duringDestruction = (_duringDestruction && managers.size() == 1);
}

// Add given element array record at idx, replacing the existing record
void CkLocMgr::insertRec(CkLocRec* rec, const CmiUInt8& id)
{
  CkLocRec* old_rec = elementNrec(id);
  hash[id] = rec;
  delete old_rec;
  noteResident(id);
}

// --- process residency table (see CkResidentTable above) ---

// Key an element by array as well as index: two arrays hand out the same ids.
CmiUInt8 CkLocMgr::residentKey(CmiUInt8 id) const
{
  return ck::ObjID(thisgroup, id).getID();
}

void CkLocMgr::noteResident(CmiUInt8 id)
{
  const CmiUInt8 key = residentKey(id);
  CkResidentTable& t = residentTable();
  std::lock_guard<std::mutex> lk(t.lock);
  auto it = t.owner.find(key);
  if (it != t.owner.end() && it->second != CkMyPe() && migDbg())
  {
    // Not fatal: an intra-process handoff can legitimately publish the arrival
    // before the source has retired its entry. Worth seeing, though, because a
    // persistent one means a departure path is not calling clearResident.
    CkPrintf("[MIG %d] resident table: id=%llu was owned by %d\n", CkMyPe(),
             (unsigned long long)id, it->second);
    fflush(stdout);
  }
  t.owner[key] = CkMyPe();
}

void CkLocMgr::clearResident(CmiUInt8 id)
{
  const CmiUInt8 key = residentKey(id);
  CkResidentTable& t = residentTable();
  std::lock_guard<std::mutex> lk(t.lock);
  auto it = t.owner.find(key);
  // Only retire our own entry. If another PE of this process has already
  // claimed the element, that claim is the newer truth and this is the losing
  // half of an intra-process handoff.
  if (it != t.owner.end() && it->second == CkMyPe()) t.owner.erase(it);
}

int CkLocMgr::residentOwnerPe(CmiUInt8 id) const
{
  const CmiUInt8 key = residentKey(id);
  CkResidentTable& t = residentTable();
  std::lock_guard<std::mutex> lk(t.lock);
  auto it = t.owner.find(key);
  return (it == t.owner.end()) ? -1 : it->second;
}

// Where a device zerocopy send should believe its target is.
//
// The transfer mode turns on one question -- is the target in my process, and
// therefore on my GPU -- and the location cache cannot answer it safely. It can
// still name a PE of this process for an element that has left, if the element
// left from another PE here; acting on that picks MEMCPY, stages nothing, and
// hands the receiver a device pointer belonging to an address space it cannot
// read. The residency table cannot be wrong about that, so it decides.
//
// Strictly more conservative than the cache alone: it only ever turns a MEMCPY
// decision into a staged one, never the reverse.
int CkLocMgr::deviceSendDestPe(const CkArrayIndex& idx)
{
  const int cached = whichPe(idx);

  CmiUInt8 id;
  if (!lookupID(idx, id)) return cached;  // index not known here at all

  const int resident = residentOwnerPe(id);
  if (resident != -1) return resident;  // in this process, exactly here

  // Not in this process. A cached PE that says otherwise is stale; report the
  // location as unconfirmed instead, which is what CkRdmaDeviceOnSender reads
  // as "stage this, I cannot promise the destination".
  if (cached != -1 && CmiNodeOf(cached) == CmiNodeOf(CkMyPe())) return -1;
  return cached;
}

size_t CkLocMgr::numResidentInProcess()
{
  CkResidentTable& t = residentTable();
  std::lock_guard<std::mutex> lk(t.lock);
  return t.owner.size();
}

// Call this on an unrecognized array index
static void abort_out_of_bounds(const CkArrayIndex& idx)
{
  CkPrintf("ERROR! Unknown array index: %s\n", idx2str(idx));
  CkAbort("Array index out of bounds\n");
}

// Look up array element in hash table.  Return NULL if not there.
CkLocRec* CkLocMgr::elementNrec(const CmiUInt8 id) const
{
  LocRecHash::const_iterator it = hash.find(id);
  return it == hash.end() ? NULL : it->second;
}

struct LocalElementCounter : public CkLocIterator
{
  unsigned int count;
  LocalElementCounter() : count(0) {}
  void addLocation(CkLocation& loc) { ++count; }
};

unsigned int CkLocMgr::numLocalElements()
{
  LocalElementCounter c;
  iterate(c);
  return c.count;
}

/********************* LocMgr: LOAD BALANCE ****************/

#if !CMK_LBDB_ON
// Empty versions of all load balancer calls
void CkLocMgr::startInserting(void) {}
void CkLocMgr::doneInserting(void) {}
#endif

#if CMK_LBDB_ON
void CkLocMgr::initLB(CkGroupID lbmgrID_, CkGroupID metalbID_)
{  // Find and register with the load balancer
  lbmgr = (LBManager*)CkLocalBranch(lbmgrID_);
  if (lbmgr == nullptr)
    CkAbort("LBManager not yet created?\n");
  DEBL((AA "Connected to load balancer %p\n" AB, lbmgr));
  if (_lb_args.metaLbOn())
  {
    the_metalb = (MetaBalancer*)CkLocalBranch(metalbID_);
    if (the_metalb == 0)
      CkAbort("MetaBalancer not yet created?\n");
  }
  syncBarrier = CkSyncBarrier::object();
  if (syncBarrier == nullptr)
    CkAbort("CkSyncBarrier not yet created?\n");
  // Register myself as an object manager
  LDOMid myId;
  myId.id = thisgroup;
  LDCallbacks myCallbacks;
  myCallbacks.migrate = (LDMigrateFn)CkLocRec::staticMigrate;
  myCallbacks.setStats = NULL;
  myCallbacks.queryEstLoad = NULL;
  myCallbacks.metaLBResumeWaitingChares =
      (LDMetaLBResumeWaitingCharesFn)CkLocRec::staticMetaLBResumeWaitingChares;
  myCallbacks.metaLBCallLBOnChares =
      (LDMetaLBCallLBOnCharesFn)CkLocRec::staticMetaLBCallLBOnChares;
  myLBHandle = lbmgr->RegisterOM(myId, this, myCallbacks);

  // Tell the lbdb that I'm registering objects
  lbmgr->RegisteringObjects(myLBHandle);

  // Set up callbacks for this LocMgr to call Registering/DoneRegistering during
  // each AtSync.
  lbBarrierBeginReceiver = syncBarrier->addBeginReceiver([=]() {
    DEBL((AA "CkLocMgr AtSync Receiver called\n" AB));
    lbmgr->RegisteringObjects(myLBHandle);
  });
  lbBarrierEndReceiver =
      syncBarrier->addEndReceiver([=]() { lbmgr->DoneRegisteringObjects(myLBHandle); });
}

void CkLocMgr::startInserting(void) { lbmgr->RegisteringObjects(myLBHandle); }
void CkLocMgr::doneInserting(void) { lbmgr->DoneRegisteringObjects(myLBHandle); }
#endif

#include "CkLocation.def.h"

/** \file cklocation.C
 *  \addtogroup CkArrayImpl
 *
 *  The location manager keeps track of an indexed set of migratable objects.
 *  It is used by the array manager to locate array elements, interact with the
 *  load balancer, and perform migrations.
 *
 *  Orion Sky Lawlor, olawlor@acm.org 9/29/2001
 */

#include "hilbert.h"
#include "partitioning_strategies.h"
#include "charm++.h"
#include "register.h"
#include "ck.h"
#include "trace.h"
#include "TopoManager.h"
#include <vector>
#include <algorithm>
#include<sstream>
#include <map>

using std::map;

#if CMK_LBDB_ON
#include "LBDatabase.h"
#include "MetaBalancer.h"
#if CMK_GLOBAL_LOCATION_UPDATE
#include "BaseLB.h"
#include "init.h"
#endif
CkpvExtern(int, _lb_obj_index);                // for lbdb user data for obj index
#endif // CMK_LBDB_ON

#ifndef CMK_CHARE_USE_PTR
CkpvExtern(int, currentChareIdx);
#endif

#if CMK_GRID_QUEUE_AVAILABLE
CpvExtern(void *, CkGridObject);
#endif

#define ARRAY_DEBUG_OUTPUT 0

#if ARRAY_DEBUG_OUTPUT 
#   define DEB(x) CkPrintf x  //General debug messages
#   define DEBI(x) CkPrintf x  //Index debug messages
#   define DEBC(x) CkPrintf x  //Construction debug messages
#   define DEBS(x) CkPrintf x  //Send/recv/broadcast debug messages
#   define DEBM(x) CkPrintf x  //Migration debug messages
#   define DEBL(x) CkPrintf x  //Load balancing debug messages
#   define DEBN(x) CkPrintf x  //Location debug messages
#   define DEBB(x) CkPrintf x  //Broadcast debug messages
#   define AA "LocMgr on %d: "
#   define AB ,CkMyPe()
#   define DEBUG(x) CkPrintf x
#   define DEBAD(x) CkPrintf x
#else
#   define DEB(X) /*CkPrintf x*/
#   define DEBI(X) /*CkPrintf x*/
#   define DEBC(X) /*CkPrintf x*/
#   define DEBS(x) /*CkPrintf x*/
#   define DEBM(X) /*CkPrintf x*/
#   define DEBL(X) /*CkPrintf x*/
#   define DEBN(x) /*CkPrintf x*/
#   define DEBB(x) /*CkPrintf x*/
#   define str(x) /**/
#   define DEBUG(x)   /**/
#   define DEBAD(x) /*CkPrintf x*/
#endif

//whether to use block mapping in the SMP node level
bool useNodeBlkMapping;

/// Message size above which the runtime will buffer messages directed at
/// unlocated array elements
int _messageBufferingThreshold;

#if CMK_LBDB_ON
/*LBDB object handles are fixed-sized, and not necc.
the same size as ArrayIndices.
*/
LDObjid idx2LDObjid(const CkArrayIndex &idx)
{
  LDObjid r;
  int i;
  const int *data=idx.data();
  if (OBJ_ID_SZ>=idx.nInts) {
    for (i=0;i<idx.nInts;i++)
      r.id[i]=data[i];
    for (i=idx.nInts;i<OBJ_ID_SZ;i++)
      r.id[i]=0;
  } else {
    //Must hash array index into LBObjid
    int j;
    for (j=0;j<OBJ_ID_SZ;j++)
    	r.id[j]=data[j];
    for (i=0;i<idx.nInts;i++)
      for (j=0;j<OBJ_ID_SZ;j++)
        r.id[j]+=circleShift(data[i],22+11*i*(j+1))+
          circleShift(data[i],21-9*i*(j+1));
  }

#if CMK_GLOBAL_LOCATION_UPDATE
  r.dimension = idx.dimension;
  r.nInts = idx.nInts; 
  r.isArrayElement = 1; 
#endif

  return r;
}

#if CMK_GLOBAL_LOCATION_UPDATE
void UpdateLocation(MigrateInfo& migData) {

  if (migData.obj.id.isArrayElement == 0) {
    return;
  }

  CkArrayIndex idx; 
  idx.dimension = migData.obj.id.dimension; 
  idx.nInts = migData.obj.id.nInts; 

  for (int i = 0; i < idx.nInts; i++) {
    idx.data()[i] = migData.obj.id.id[i];    
  }

  CkGroupID locMgrGid;
  locMgrGid.idx = migData.obj.id.locMgrGid;
  CkLocMgr *localLocMgr = (CkLocMgr *) CkLocalBranch(locMgrGid);
  localLocMgr->updateLocation(idx, migData.to_pe); 
}
#endif

#endif

/*********************** Array Messages ************************/
CmiUInt8 CkArrayMessage::array_element_id(void)
{
  return ck::ObjID(UsrToEnv((void *)this)->getRecipientID()).getElementID();
}
unsigned short &CkArrayMessage::array_ep(void)
{
	return UsrToEnv((void *)this)->getsetArrayEp();
}
unsigned short &CkArrayMessage::array_ep_bcast(void)
{
	return UsrToEnv((void *)this)->getsetArrayBcastEp();
}
unsigned char &CkArrayMessage::array_hops(void)
{
	return UsrToEnv((void *)this)->getsetArrayHops();
}
unsigned int CkArrayMessage::array_getSrcPe(void)
{
	return UsrToEnv((void *)this)->getsetArraySrcPe();
}
unsigned int CkArrayMessage::array_ifNotThere(void)
{
	return UsrToEnv((void *)this)->getArrayIfNotThere();
}
void CkArrayMessage::array_setIfNotThere(unsigned int i)
{
	UsrToEnv((void *)this)->setArrayIfNotThere(i);
}

/*********************** Array Map ******************
Given an array element index, an array map tells us 
the index's "home" Pe.  This is the Pe the element will
be created on, and also where messages to this element will
be forwarded by default.
*/

CkArrayMap::CkArrayMap(void) { }
CkArrayMap::~CkArrayMap() { }
int CkArrayMap::registerArray(const CkArrayIndex& numElements, CkArrayID aid)
{return 0;}
void CkArrayMap::unregisterArray(int idx)
{ }

#define CKARRAYMAP_POPULATE_INITIAL(POPULATE_CONDITION) \
int i; \
int index[6]; \
int start_data[6], end_data[6], step_data[6]; \
for (int d = 0; d < 6; d++) { \
  start_data[d] = 0; \
  end_data[d] = step_data[d] = 1; \
  if (end.dimension >= 4 && d < end.dimension) { \
    start_data[d] = ((short int*)start.data())[d]; \
    end_data[d] = ((short int*)end.data())[d]; \
    step_data[d] = ((short int*)step.data())[d]; \
  } else if (d < end.dimension) { \
    start_data[d] = start.data()[d]; \
    end_data[d] = end.data()[d]; \
    step_data[d] = step.data()[d]; \
  } \
} \
 \
for (index[0] = start_data[0]; index[0] < end_data[0]; index[0] += step_data[0]) { \
  for (index[1] = start_data[1]; index[1] < end_data[1]; index[1] += step_data[1]) { \
    for (index[2] = start_data[2]; index[2] < end_data[2]; index[2] += step_data[2]) { \
      for (index[3] = start_data[3]; index[3] < end_data[3]; index[3] += step_data[3]) { \
        for (index[4] = start_data[4]; index[4] < end_data[4]; index[4] += step_data[4]) { \
          for (index[5] = start_data[5]; index[5] < end_data[5]; index[5] += step_data[5]) { \
            if (end.dimension == 1) { \
              i = index[0]; \
              CkArrayIndex1D idx(index[0]); \
              if (POPULATE_CONDITION) { \
                mgr->insertInitial(idx,CkCopyMsg(&ctorMsg)); \
              } \
            } else if (end.dimension == 2) { \
              i = index[0] * end_data[1] + index[1]; \
              CkArrayIndex2D idx(index[0], index[1]); \
              if (POPULATE_CONDITION) { \
                mgr->insertInitial(idx,CkCopyMsg(&ctorMsg)); \
              } \
            } else if (end.dimension == 3) { \
              i = (index[0]*end_data[1] + index[1]) * end_data[2] + index[2]; \
              CkArrayIndex3D idx(index[0], index[1], index[2]); \
              if (POPULATE_CONDITION) { \
                mgr->insertInitial(idx,CkCopyMsg(&ctorMsg)); \
              } \
            } else if (end.dimension == 4) { \
              i = ((index[0]*end_data[1] + index[1]) * end_data[2] + index[2]) * end_data[3] + index[3]; \
              CkArrayIndex4D idx(index[0], index[1], index[2], index[3]); \
              if (POPULATE_CONDITION) { \
                mgr->insertInitial(idx,CkCopyMsg(&ctorMsg)); \
              } \
            } else if (end.dimension == 5) { \
              i = (((index[0]*end_data[1] + index[1]) * end_data[2] + index[2]) * end_data[3] + index[3]) * end_data[4] + index[4]; \
              CkArrayIndex5D idx(index[0], index[1], index[2], index[3], index[4]); \
              if (POPULATE_CONDITION) { \
                mgr->insertInitial(idx,CkCopyMsg(&ctorMsg)); \
              } \
            } else if (end.dimension == 6) { \
              i = ((((index[0]*end_data[1] + index[1]) * end_data[2] + index[2]) * end_data[3] + index[3]) * end_data[4] + index[4]) * end_data[5] + index[5]; \
              CkArrayIndex6D idx(index[0], index[1], index[2], index[3], index[4], index[5]); \
              if (POPULATE_CONDITION) { \
                mgr->insertInitial(idx,CkCopyMsg(&ctorMsg)); \
              } \
            } \
          } \
        } \
      } \
    } \
  } \
}
  
void CkArrayMap::populateInitial(int arrayHdl,CkArrayOptions& options,void *ctorMsg,CkArray *mgr)
{
  CkArrayIndex start = options.getStart();
  CkArrayIndex end = options.getEnd();
  CkArrayIndex step = options.getStep();
	if (end.nInts==0) {
          CkFreeMsg(ctorMsg);
          return;
        }
	int thisPe=CkMyPe();
        /* The CkArrayIndex is supposed to have at most 3 dimensions, which
           means that all the fields are ints, and numElements.nInts represents
           how many of them are used */
        CKARRAYMAP_POPULATE_INITIAL(procNum(arrayHdl,idx)==thisPe);

#if CMK_BIGSIM_CHARM
        BgEntrySplit("split-array-new-end");
#endif

	mgr->doneInserting();
	CkFreeMsg(ctorMsg);
}

CkGroupID _defaultArrayMapID;
CkGroupID _fastArrayMapID;

class RRMap : public CkArrayMap
{
public:
  RRMap(void)
  {
    DEBC((AA "Creating RRMap\n" AB));
  }
  RRMap(CkMigrateMessage *m):CkArrayMap(m){}
  int procNum(int /*arrayHdl*/, const CkArrayIndex &i)
  {
#if 1
    if (i.dimension==1) {
      //Map 1D integer indices in simple round-robin fashion
      int ans = (i.data()[0])%CkNumPes();
      while(!CmiNodeAlive(ans) || (ans == CkMyPe() && CkpvAccess(startedEvac))){
        ans = (ans+1)%CkNumPes();
      }
      return ans;
    }
    else 
#endif
    {
	//Map other indices based on their hash code, mod a big prime.
	unsigned int hash=(i.hash()+739)%1280107;
	int ans = (hash % CkNumPes());
	while(!CmiNodeAlive(ans)){
	  ans = (ans+1)%CkNumPes();
	}
	return ans;
    }
  }
};

/** 
 * Class used to store the dimensions of the array and precalculate numChares,
 * binSize and other values for the DefaultArrayMap -- ASB
 */
class arrayMapInfo {
public:
  CkArrayIndex _nelems;
  int _binSizeFloor;		/* floor of numChares/numPes */
  int _binSizeCeil;		/* ceiling of numChares/numPes */
  int _numChares;		/* initial total number of chares */
  int _remChares;		/* numChares % numPes -- equals the number of
				   processors in the first set */
  int _numFirstSet;		/* _remChares X (_binSize + 1) -- number of
				   chares in the first set */

  int _nBinSizeFloor;           /* floor of numChares/numNodes */
  int _nRemChares;              /* numChares % numNodes -- equals the number of
                                   nodes in the first set */
  int _nNumFirstSet;            /* _remChares X (_binSize + 1) -- number of
                                   chares in the first set of nodes */

  /** All processors are divided into two sets. Processors in the first set
   *  have one chare more than the processors in the second set. */

  arrayMapInfo(void) { }

  arrayMapInfo(const CkArrayIndex& n) : _nelems(n), _numChares(0) {
    compute_binsize();
  }

  ~arrayMapInfo() {}
  
  void compute_binsize()
  {
    int numPes = CkNumPes();
    //Now assuming homogenous nodes where each node has the same number of PEs
    int numNodes = CkNumNodes();

    if (_nelems.dimension == 1) {
      _numChares = _nelems.data()[0];
    } else if (_nelems.dimension == 2) {
      _numChares = _nelems.data()[0] * _nelems.data()[1];
    } else if (_nelems.dimension == 3) {
      _numChares = _nelems.data()[0] * _nelems.data()[1] * _nelems.data()[2];
    } else if (_nelems.dimension == 4) {
      _numChares = (int)(((short int*)_nelems.data())[0] * ((short int*)_nelems.data())[1] * ((short int*)_nelems.data())[2] *
                   ((short int*)_nelems.data())[3]);
    } else if (_nelems.dimension == 5) {
      _numChares = (int)(((short int*)_nelems.data())[0] * ((short int*)_nelems.data())[1] * ((short int*)_nelems.data())[2] *
                   ((short int*)_nelems.data())[3] * ((short int*)_nelems.data())[4]);
    } else if (_nelems.dimension == 6) {
      _numChares = (int)(((short int*)_nelems.data())[0] * ((short int*)_nelems.data())[1] * ((short int*)_nelems.data())[2] *
                   ((short int*)_nelems.data())[3] * ((short int*)_nelems.data())[4] * ((short int*)_nelems.data())[5]);
    }

    _remChares = _numChares % numPes;
    _binSizeFloor = (int)floor((double)_numChares/(double)numPes);
    _binSizeCeil = (int)ceil((double)_numChares/(double)numPes);
    _numFirstSet = _remChares * (_binSizeFloor + 1);

    _nRemChares = _numChares % numNodes;
    _nBinSizeFloor = _numChares/numNodes;
    _nNumFirstSet = _nRemChares * (_nBinSizeFloor +1);
  }

  void pup(PUP::er& p){
    p|_nelems;
    p|_binSizeFloor;
    p|_binSizeCeil;
    p|_numChares;
    p|_remChares;
    p|_numFirstSet;
    p|_nBinSizeFloor;
    p|_nRemChares;
    p|_nNumFirstSet;
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
  DefaultArrayMap(void) {
    DEBC((AA "Creating DefaultArrayMap\n" AB));
  }

  DefaultArrayMap(CkMigrateMessage *m) : RRMap(m){}

  int registerArray(const CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx = amaps.size();
    amaps.resize(idx+1);
    amaps[idx] = new arrayMapInfo(numElements);
    return idx;
  }

  void unregisterArray(int idx)
  {
    delete amaps[idx];
    amaps[idx] = NULL;
  }
 
  int procNum(int arrayHdl, const CkArrayIndex &i) {
    int flati;
    if (amaps[arrayHdl]->_nelems.dimension == 0) {
      return RRMap::procNum(arrayHdl, i);
    }

    if (i.dimension == 1) {
      flati = i.data()[0];
    } else if (i.dimension == 2) {
      flati = i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1];
    } else if (i.dimension == 3) {
      flati = (i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1]) *
              amaps[arrayHdl]->_nelems.data()[2] + i.data()[2];
    } else if (i.dimension == 4) {
      flati = (int)(((((short int*)i.data())[0] * ((short int*)amaps[arrayHdl]->_nelems.data())[1] + ((short int*)i.data())[1]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[2] + ((short int*)i.data())[2]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[3] + ((short int*)i.data())[3]);
    } else if (i.dimension == 5) {
      flati = (int)((((((short int*)i.data())[0] * ((short int*)amaps[arrayHdl]->_nelems.data())[1] + ((short int*)i.data())[1]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[2] + ((short int*)i.data())[2]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[3] + ((short int*)i.data())[3]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[4] + ((short int*)i.data())[4]);
    } else if (i.dimension == 6) {
      flati = (int)(((((((short int*)i.data())[0] * ((short int*)amaps[arrayHdl]->_nelems.data())[1] + ((short int*)i.data())[1]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[2] + ((short int*)i.data())[2]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[3] + ((short int*)i.data())[3]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[4] + ((short int*)i.data())[4]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[5] + ((short int*)i.data())[5]);
    }
#if CMK_ERROR_CHECKING
    else {
      CkAbort("CkArrayIndex has more than 6 dimensions!");
    }
#endif

    if(useNodeBlkMapping){
      if(flati < amaps[arrayHdl]->_numChares){
        int numCharesOnNode = amaps[arrayHdl]->_nBinSizeFloor;
        int startNodeID, offsetInNode;
        if(flati < amaps[arrayHdl]->_nNumFirstSet){
          numCharesOnNode++;
          startNodeID = flati/numCharesOnNode;
          offsetInNode = flati%numCharesOnNode;
        }else{
          startNodeID = amaps[arrayHdl]->_nRemChares+(flati-amaps[arrayHdl]->_nNumFirstSet)/numCharesOnNode;
          offsetInNode = (flati-amaps[arrayHdl]->_nNumFirstSet)%numCharesOnNode;
        }
        int nodeSize = CkMyNodeSize(); //assuming every node has same number of PEs
        int elemsPerPE = numCharesOnNode/nodeSize;
        int remElems = numCharesOnNode%nodeSize;
        int firstSetPEs = remElems*(elemsPerPE+1);
        if(offsetInNode<firstSetPEs){
          return CkNodeFirst(startNodeID)+offsetInNode/(elemsPerPE+1);
        }else{
          return CkNodeFirst(startNodeID)+remElems+(offsetInNode-firstSetPEs)/elemsPerPE;
        }
      } else
          return (flati % CkNumPes());
    }
    //regular PE-based block mapping
    if(flati < amaps[arrayHdl]->_numFirstSet)
      return (flati / (amaps[arrayHdl]->_binSizeFloor + 1));
    else if (flati < amaps[arrayHdl]->_numChares)
      return (amaps[arrayHdl]->_remChares + (flati - amaps[arrayHdl]->_numFirstSet) / (amaps[arrayHdl]->_binSizeFloor));
    else
      return (flati % CkNumPes());
  }

  void pup(PUP::er& p){
    RRMap::pup(p);
    int npes = CkNumPes();
    p|npes;
    p|amaps;
    if (p.isUnpacking() && npes != CkNumPes())  {   // binSize needs update
      for (int i=0; i<amaps.size(); i++)
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
  FastArrayMap(void) {
    DEBC((AA "Creating FastArrayMap\n" AB));
  }

  FastArrayMap(CkMigrateMessage *m) : DefaultArrayMap(m){}

  int registerArray(const CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx;
    idx = DefaultArrayMap::registerArray(numElements, aid);

    return idx;
  }

  int procNum(int arrayHdl, const CkArrayIndex &i) {
    int flati = 0;
    if (amaps[arrayHdl]->_nelems.dimension == 0) {
      return RRMap::procNum(arrayHdl, i);
    }

    if (i.dimension == 1) {
      flati = i.data()[0];
    } else if (i.dimension == 2) {
      flati = i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1];
    } else if (i.dimension == 3) {
      flati = (i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1]) *
              amaps[arrayHdl]->_nelems.data()[2] + i.data()[2];
    } else if (i.dimension == 4) {
      flati = (int)(((((short int*)i.data())[0] * ((short int*)amaps[arrayHdl]->_nelems.data())[1] + ((short int*)i.data())[1]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[2] + ((short int*)i.data())[2]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[3] + ((short int*)i.data())[3]);
    } else if (i.dimension == 5) {
      flati = (int)((((((short int*)i.data())[0] * ((short int*)amaps[arrayHdl]->_nelems.data())[1] + ((short int*)i.data())[1]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[2] + ((short int*)i.data())[2]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[3] + ((short int*)i.data())[3]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[4] + ((short int*)i.data())[4]);
    } else if (i.dimension == 6) {
      flati = (int)(((((((short int*)i.data())[0] * ((short int*)amaps[arrayHdl]->_nelems.data())[1] + ((short int*)i.data())[1]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[2] + ((short int*)i.data())[2]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[3] + ((short int*)i.data())[3]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[4] + ((short int*)i.data())[4]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[5] + ((short int*)i.data())[5]);
    }
#if CMK_ERROR_CHECKING
    else {
      CkAbort("CkArrayIndex has more than 6 dimensions!");
    }
#endif

    /** binSize used in DefaultArrayMap is the floor of numChares/numPes
     *  but for this FastArrayMap, we need the ceiling */
    return (flati / amaps[arrayHdl]->_binSizeCeil);
  }

  void pup(PUP::er& p){
    DefaultArrayMap::pup(p);
  }
};

/* *
 * Hilbert map object -- This does hilbert mapping.
 * Convert array indices into 1D fashion according to their Hilbert filling curve 
 */

typedef struct {
    int intIndex;
    std::vector<int> coords;
}hilbert_pair;

bool operator== ( hilbert_pair p1, hilbert_pair p2) 
{
    return p1.intIndex == p2.intIndex;
}

bool myCompare(hilbert_pair p1, hilbert_pair p2)
{
    return p1.intIndex < p2.intIndex;
}

class HilbertArrayMap: public DefaultArrayMap
{
  std::vector<int> allpairs;
  int *procList;
public:
  HilbertArrayMap(void) {
    procList = new int[CkNumPes()]; 
    getHilbertList(procList);
    DEBC((AA "Creating HilbertArrayMap\n" AB));
  }

  HilbertArrayMap(CkMigrateMessage *m) : DefaultArrayMap(m){}

  ~HilbertArrayMap()
  {
    if(procList)
      delete []procList;
  }

  int registerArray(const CkArrayIndex& i, CkArrayID aid)
  {
    int idx;
    idx = DefaultArrayMap::registerArray(i, aid);
   
    if (i.dimension == 1) {
      //CkPrintf("1D %d\n", amaps[idx]->_nelems.data()[0]); 
    } else if (i.dimension == 2) {
      //CkPrintf("2D %d:%d\n", amaps[idx]->_nelems.data()[0], amaps[idx]->_nelems.data()[1]); 
      const int dims = 2;
      int nDim0 = amaps[idx]->_nelems.data()[0];
      int nDim1 = amaps[idx]->_nelems.data()[1];
      int index;
      int counter = 0;
      std::vector<int> coords;
      allpairs.resize(nDim0*nDim1);
      coords.resize(dims);
      for(int i=0; i<nDim0; i++)
        for(int j=0; j<nDim1; j++)
        {
          coords[0] = i;
          coords[1] = j;
          index = Hilbert_to_int( coords, dims);
          //CkPrintf("(%d:%d)----------> %d \n", i, j, index);
          allpairs[counter] = index;
          counter++;
        }
    } else if (i.dimension == 3) {
      //CkPrintf("3D %d:%d:%d\n", amaps[idx]->_nelems.data()[0], amaps[idx]->_nelems.data()[1],
      //        amaps[idx]->_nelems.data()[2]);
      const int dims = 3;
      int nDim0 = amaps[idx]->_nelems.data()[0];
      int nDim1 = amaps[idx]->_nelems.data()[1];
      int nDim2 = amaps[idx]->_nelems.data()[2];
      int index;
      int counter = 0;
      std::vector<int> coords;
      allpairs.resize(nDim0*nDim1*nDim2);
      coords.resize(dims);
      for(int i=0; i<nDim0; i++)
        for(int j=0; j<nDim1; j++)
          for(int k=0; k<nDim2; k++)
          {
            coords[0] = i;
            coords[1] = j;
            coords[2] = k;
            index = Hilbert_to_int( coords, dims);
            allpairs[counter] = index;
            counter++;
          }
    } else if (i.dimension == 4) {
      //CkPrintf("4D %hd:%hd:%hd:%hd\n", ((short int*)amaps[idx]->_nelems.data())[0],
      //        ((short int*)amaps[idx]->_nelems.data())[1], ((short int*)amaps[idx]->_nelems.data())[2],
      //        ((short int*)amaps[idx]->_nelems.data())[3]);
      const int dims = 4;
      int nDim[dims];
      for(int k=0; k<dims; k++) {
        nDim[k] = (int)((short int*)amaps[idx]->_nelems.data())[k];
      }
      int index;
      int counter = 0;
      std::vector<int> coords;
      allpairs.resize(nDim[0]*nDim[1]*nDim[2]*nDim[3]);
      coords.resize(dims);
      for(int i=0; i<nDim[0]; i++)
        for(int j=0; j<nDim[1]; j++)
          for(int k=0; k<nDim[2]; k++)
            for(int x=0; x<nDim[3]; x++)
            {
              coords[0] = i;
              coords[1] = j;
              coords[2] = k;
              coords[3] = x;
              index = Hilbert_to_int(coords, dims);
              allpairs[counter] = index;
              counter++;
            }
    } else if (i.dimension == 5) {
      //CkPrintf("5D %hd:%hd:%hd:%hd:%hd\n", ((short int*)amaps[idx]->_nelems.data())[0],
      //        ((short int*)amaps[idx]->_nelems.data())[1], ((short int*)amaps[idx]->_nelems.data())[2],
      //        ((short int*)amaps[idx]->_nelems.data())[3], ((short int*)amaps[idx]->_nelems.data())[4]);
      const int dims = 5;
      int nDim[dims];
      for(int k=0; k<dims; k++) {
        nDim[k] = (int)((short int*)amaps[idx]->_nelems.data())[k];
      }
      int index;
      int counter = 0;
      std::vector<int> coords;
      allpairs.resize(nDim[0]*nDim[1]*nDim[2]*nDim[3]*nDim[4]);
      coords.resize(dims);
      for(int i=0; i<nDim[0]; i++)
        for(int j=0; j<nDim[1]; j++)
          for(int k=0; k<nDim[2]; k++)
            for(int x=0; x<nDim[3]; x++)
              for(int y=0; y<nDim[4]; y++)
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
    } else if (i.dimension == 6) {
      //CkPrintf("6D %hd:%hd:%hd:%hd:%hd:%hd\n", ((short int*)amaps[idx]->_nelems.data())[0],
      //        ((short int*)amaps[idx]->_nelems.data())[1], ((short int*)amaps[idx]->_nelems.data())[2],
      //        ((short int*)amaps[idx]->_nelems.data())[3], ((short int*)amaps[idx]->_nelems.data())[4],
      //        ((short int*)amaps[idx]->_nelems.data())[5]);
      const int dims = 6;
      int nDim[dims];
      for(int k=0; k<dims; k++) {
        nDim[k] = (int)((short int*)amaps[idx]->_nelems.data())[k];
      }
      int index;
      int counter = 0;
      std::vector<int> coords;
      allpairs.resize(nDim[0]*nDim[1]*nDim[2]*nDim[3]*nDim[4]*nDim[5]);
      coords.resize(dims);
      for(int i=0; i<nDim[0]; i++)
        for(int j=0; j<nDim[1]; j++)
          for(int k=0; k<nDim[2]; k++)
            for(int x=0; x<nDim[3]; x++)
              for(int y=0; y<nDim[4]; y++)
                for(int z=0; z<nDim[5]; z++)
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

  int procNum(int arrayHdl, const CkArrayIndex &i) {
    int flati = 0;
    int myInt;
    int dest;
    if (amaps[arrayHdl]->_nelems.dimension == 0) {
      return RRMap::procNum(arrayHdl, i);
    }
    if (i.dimension == 1) {
      flati = i.data()[0];
    } else if (i.dimension == 2) {
      int nDim1 = amaps[arrayHdl]->_nelems.data()[1];
      myInt = i.data()[0] * nDim1 + i.data()[1];
      flati = allpairs[myInt];
    } else if (i.dimension == 3) {
      hilbert_pair mypair;
      mypair.coords.resize(3);
      int nDim[2];
      for (int j = 0; j < 2; j++) {
        nDim[j] = amaps[arrayHdl]->_nelems.data()[j+1];
      }
      myInt = i.data()[0] * nDim[0] * nDim[1] + i.data()[1] * nDim[1] + i.data()[2];
      flati = allpairs[myInt];
    } else if (i.dimension == 4) {
      hilbert_pair mypair;
      mypair.coords.resize(4);
      short int nDim[3];
      for (int j = 0; j < 3; j++) {
        nDim[j] = ((short int*)amaps[arrayHdl]->_nelems.data())[j+1];
      }
      myInt = (int)(((short int*)i.data())[0] * nDim[0] * nDim[1] * nDim[2] +
              ((short int*)i.data())[1] * nDim[1] * nDim[2] +
              ((short int*)i.data())[2] * nDim[2] +
              ((short int*)i.data())[3]);
      flati = allpairs[myInt];
    } else if (i.dimension == 5) {
      hilbert_pair mypair;
      mypair.coords.resize(5);
      short int nDim[4];
      for (int j = 0; j < 4; j++) {
        nDim[j] = ((short int*)amaps[arrayHdl]->_nelems.data())[j+1];
      }
      myInt = (int)(((short int*)i.data())[0] * nDim[0] * nDim[1] * nDim[2] * nDim[3] +
              ((short int*)i.data())[1] * nDim[1] * nDim[2] * nDim[3] +
              ((short int*)i.data())[2] * nDim[2] * nDim[3] +
              ((short int*)i.data())[3] * nDim[3] +
              ((short int*)i.data())[4]);
      flati = allpairs[myInt];
    } else if (i.dimension == 6) {
      hilbert_pair mypair;
      mypair.coords.resize(6);
      short int nDim[5];
      for (int j = 0; j < 5; j++) {
        nDim[j] = ((short int*)amaps[arrayHdl]->_nelems.data())[j+1];
      }
      myInt = (int)(((short int*)i.data())[0] * nDim[0] * nDim[1] * nDim[2] * nDim[3] * nDim[4] +
              ((short int*)i.data())[1] * nDim[1] * nDim[2] * nDim[3] * nDim[4] +
              ((short int*)i.data())[2] * nDim[2] * nDim[3] * nDim[4] +
              ((short int*)i.data())[3] * nDim[3] * nDim[4] +
              ((short int*)i.data())[4] * nDim[4] +
              ((short int*)i.data())[5]);
      flati = allpairs[myInt];
    }
#if CMK_ERROR_CHECKING
    else {
      CkAbort("CkArrayIndex has more than 6 dimensions!");
    }
#endif

    /** binSize used in DefaultArrayMap is the floor of numChares/numPes
     *  but for this FastArrayMap, we need the ceiling */
    int block = flati / amaps[arrayHdl]->_binSizeCeil;
    //for(int i=0; i<CkNumPes(); i++)
    //    CkPrintf("(%d:%d) ", i, procList[i]);
    //CkPrintf("\n");
    //CkPrintf("block [%d:%d]\n", block, procList[block]);
    return procList[block];
  }

  void pup(PUP::er& p){
    DefaultArrayMap::pup(p);
  }
};


/**
 * This map can be used for topology aware mapping when the mapping is provided
 * through a file -- ASB
 */
class ReadFileMap : public DefaultArrayMap
{
private:
  CkVec<int> mapping;

public:
  ReadFileMap(void) {
    DEBC((AA "Creating ReadFileMap\n" AB));
  }

  ReadFileMap(CkMigrateMessage *m) : DefaultArrayMap(m){}

  int registerArray(const CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx;
    idx = DefaultArrayMap::registerArray(numElements, aid);

    if(mapping.size() == 0) {
      int numChares;

      if (amaps[idx]->_nelems.dimension == 1) {
        numChares = amaps[idx]->_nelems.data()[0];
      } else if (amaps[idx]->_nelems.dimension == 2) {
        numChares = amaps[idx]->_nelems.data()[0] * amaps[idx]->_nelems.data()[1];
      } else if (amaps[idx]->_nelems.dimension == 3) {
        numChares = amaps[idx]->_nelems.data()[0] * amaps[idx]->_nelems.data()[1] *
                    amaps[idx]->_nelems.data()[2];
      } else if (amaps[idx]->_nelems.dimension == 4) {
        numChares = (int)(((short int*)amaps[idx]->_nelems.data())[0] * ((short int*)amaps[idx]->_nelems.data())[1] *
                    ((short int*)amaps[idx]->_nelems.data())[2] * ((short int*)amaps[idx]->_nelems.data())[3]);
      } else if (amaps[idx]->_nelems.dimension == 5) {
        numChares = (int)(((short int*)amaps[idx]->_nelems.data())[0] * ((short int*)amaps[idx]->_nelems.data())[1] *
                    ((short int*)amaps[idx]->_nelems.data())[2] * ((short int*)amaps[idx]->_nelems.data())[3] *
                    ((short int*)amaps[idx]->_nelems.data())[4]);
      } else if (amaps[idx]->_nelems.dimension == 6) {
        numChares = (int)(((short int*)amaps[idx]->_nelems.data())[0] * ((short int*)amaps[idx]->_nelems.data())[1] *
                    ((short int*)amaps[idx]->_nelems.data())[2] * ((short int*)amaps[idx]->_nelems.data())[3] *
                    ((short int*)amaps[idx]->_nelems.data())[4] * ((short int*)amaps[idx]->_nelems.data())[5]);
      } else {
        CkAbort("CkArrayIndex has more than 6 dimension!");
      }

      mapping.resize(numChares);
      FILE *mapf = fopen("mapfile", "r");
      TopoManager tmgr;
      int x, y, z, t;

      for(int i=0; i<numChares; i++) {
        (void) fscanf(mapf, "%d %d %d %d", &x, &y, &z, &t);
        mapping[i] = tmgr.coordinatesToRank(x, y, z, t);
      }
      fclose(mapf);
    }

    return idx;
  }

  int procNum(int arrayHdl, const CkArrayIndex &i) {
    int flati;

    if (i.dimension == 1) {
      flati = i.data()[0];
    } else if (i.dimension == 2) {
      flati = i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1];
    } else if (i.dimension == 3) {
      flati = (i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1]) *
              amaps[arrayHdl]->_nelems.data()[2] + i.data()[2];
    } else if (i.dimension == 4) {
      flati = (int)(((((short int*)i.data())[0] * ((short int*)amaps[arrayHdl]->_nelems.data())[1] + ((short int*)i.data())[1]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[2] + ((short int*)i.data())[2]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[3] + ((short int*)i.data())[3]);
    } else if (i.dimension == 5) {
      flati = (int)((((((short int*)i.data())[0] * ((short int*)amaps[arrayHdl]->_nelems.data())[1] + ((short int*)i.data())[1]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[2] + ((short int*)i.data())[2]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[3] + ((short int*)i.data())[3]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[4] + ((short int*)i.data())[4]);
    } else if (i.dimension == 6) {
      flati = (int)(((((((short int*)i.data())[0] * ((short int*)amaps[arrayHdl]->_nelems.data())[1] + ((short int*)i.data())[1]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[2] + ((short int*)i.data())[2]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[3] + ((short int*)i.data())[3]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[4] + ((short int*)i.data())[4]) *
              ((short int*)amaps[arrayHdl]->_nelems.data())[5] + ((short int*)i.data())[5]);
    } else {
      CkAbort("CkArrayIndex has more than 6 dimensions!");
    }

    return mapping[flati];
  }

  void pup(PUP::er& p){
    DefaultArrayMap::pup(p);
    p|mapping;
  }
};

class BlockMap : public RRMap
{
public:
  BlockMap(void){
    DEBC((AA "Creating BlockMap\n" AB));
  }
  BlockMap(CkMigrateMessage *m):RRMap(m){ }
  void populateInitial(int arrayHdl,CkArrayOptions& options,void *ctorMsg,CkArray *mgr){
    CkArrayIndex start = options.getStart();
    CkArrayIndex end = options.getEnd();
    CkArrayIndex step = options.getStep();
    if (end.dimension == 0) {
      CkFreeMsg(ctorMsg);
      return;
    }
    int thisPe=CkMyPe();
    int numPes=CkNumPes();
    int binSize;
    if (end.dimension == 1) {
      binSize = (int)ceil((double)(end.data()[0]) / (double)numPes);
    } else if (end.dimension == 2) {
      binSize = (int)ceil((double)(end.data()[0] * end.data()[1]) / (double)numPes);
    } else if (end.dimension == 3) {
      binSize = (int)ceil((double)(end.data()[0] * end.data()[1] * end.data()[2])) / (double)numPes;
    } else if (end.dimension == 4) {
      binSize = (int)ceil((double)(((short int*)end.data())[0] * ((short int*)end.data())[1] *
                ((short int*)end.data())[2] * ((short int*)end.data())[3]) / (double)numPes);
    } else if (end.dimension == 5) {
      binSize = (int)ceil((double)(((short int*)end.data())[0] * ((short int*)end.data())[1] *
                ((short int*)end.data())[2] * ((short int*)end.data())[3] * ((short int*)end.data())[4]) /
                (double)numPes);
    } else if (end.dimension == 6) {
      binSize = (int)ceil((double)(((short int*)end.data())[0] * ((short int*)end.data())[1] *
                ((short int*)end.data())[2] * ((short int*)end.data())[3] * ((short int*)end.data())[4] *
                ((short int*)end.data())[5]) / (double)numPes);
    } else {
      CkAbort("CkArrayIndex has more than 6 dimensions!");
    }
    CKARRAYMAP_POPULATE_INITIAL(i/binSize==thisPe);

    /*CkArrayIndex idx;
    for (idx=numElements.begin(); idx<numElements; idx.getNext(numElements)) {
      int binSize = (int)ceil((double)numElements.getCombinedCount()/(double)numPes);
      if (i/binSize==thisPe)
        mgr->insertInitial(idx,CkCopyMsg(&ctorMsg));
    }*/
    mgr->doneInserting();
    CkFreeMsg(ctorMsg);
  }
};

/**
 * map object-- use seed load balancer.  
 */
class CldMap : public CkArrayMap
{
public:
  CldMap(void)
  {
	  DEBC((AA "Creating CldMap\n" AB));
  }
  CldMap(CkMigrateMessage *m):CkArrayMap(m){}
  int homePe(int /*arrayHdl*/, const CkArrayIndex &i)
  {
    if (i.dimension == 1) {
      //Map 1D integer indices in simple round-robin fashion
      return (i.data()[0])%CkNumPes();
    }
    else 
      {
	//Map other indices based on their hash code, mod a big prime.
	unsigned int hash=(i.hash()+739)%1280107;
	return (hash % CkNumPes());
      }
  }
  int procNum(int arrayHdl, const CkArrayIndex &i)
  {
     return CLD_ANYWHERE;   // -1
  }
  void populateInitial(int arrayHdl,CkArrayOptions& options,void *ctorMsg,CkArray *mgr)  {
        CkArrayIndex start = options.getStart();
        CkArrayIndex end = options.getEnd();
        CkArrayIndex step = options.getStep();
        if (end.dimension == 0) {
          CkFreeMsg(ctorMsg);
          return;
        }
        int thisPe=CkMyPe();
        int numPes=CkNumPes();

        CKARRAYMAP_POPULATE_INITIAL(i%numPes==thisPe);
        mgr->doneInserting();
        CkFreeMsg(ctorMsg);
  }
};


/// A class responsible for parsing the command line arguments for the PE
/// to extract the format string passed in with +ConfigurableRRMap
class ConfigurableRRMapLoader {
public:
  
  int *locations;
  int objs_per_block;
  int PE_per_block;

  /// labels for states used when parsing the ConfigurableRRMap from ARGV
  enum ConfigurableRRMapLoadStatus{
    not_loaded,
    loaded_found,
    loaded_not_found
  };
  
  enum ConfigurableRRMapLoadStatus state;
  
  ConfigurableRRMapLoader(){
    state = not_loaded;
    locations = NULL;
    objs_per_block = 0;
    PE_per_block = 0;
  }
  
  /// load configuration if possible, and return whether a valid configuration exists
  bool haveConfiguration() {
    if(state == not_loaded) {
      DEBUG(("[%d] loading ConfigurableRRMap configuration\n", CkMyPe()));
      char **argv=CkGetArgv();
      char *configuration = NULL;
      bool found = CmiGetArgString(argv, "+ConfigurableRRMap", &configuration);
      if(!found){
	DEBUG(("Couldn't find +ConfigurableRRMap command line argument\n"));
	state = loaded_not_found;
	return false;
      } else {

	DEBUG(("Found +ConfigurableRRMap command line argument in %p=\"%s\"\n", configuration, configuration));

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
	locations = new int[objs_per_block];
	for(int i=0;i<objs_per_block;i++){
	  locations[i] = 0;
	  CkAssert(instream.good());
	  instream >> locations[i];
	  CkAssert(locations[i] < PE_per_block);
	}
	state = loaded_found;
	return true;
      }

    } else {
      DEBUG(("[%d] ConfigurableRRMap has already been loaded\n", CkMyPe()));
      return state == loaded_found;
    }      
     
  }
  
};

CkpvDeclare(ConfigurableRRMapLoader, myConfigRRMapState);

void _initConfigurableRRMap(){
  CkpvInitialize(ConfigurableRRMapLoader, myConfigRRMapState);
}


/// Try to load the command line arguments for ConfigurableRRMap
bool haveConfigurableRRMap(){
  DEBUG(("haveConfigurableRRMap()\n"));
  ConfigurableRRMapLoader &loader =  CkpvAccess(myConfigRRMapState);
  return loader.haveConfiguration();
}

class ConfigurableRRMap : public RRMap
{
public:
  ConfigurableRRMap(void){
	DEBC((AA "Creating ConfigurableRRMap\n" AB));
  }
  ConfigurableRRMap(CkMigrateMessage *m):RRMap(m){ }


  void populateInitial(int arrayHdl,CkArrayOptions& options,void *ctorMsg,CkArray *mgr){
    CkArrayIndex start = options.getStart();
    CkArrayIndex end = options.getEnd();
    CkArrayIndex step = options.getStep();
    // Try to load the configuration from command line argument
    CkAssert(haveConfigurableRRMap());
    ConfigurableRRMapLoader &loader =  CkpvAccess(myConfigRRMapState);
    if (end.dimension == 0) {
      CkFreeMsg(ctorMsg);
      return;
    }
    int thisPe=CkMyPe();
    int maxIndex = end.data()[0];
    DEBUG(("[%d] ConfigurableRRMap: index=%d,%d,%d\n", CkMyPe(),(int)end.data()[0], (int)end.data()[1], (int)end.data()[2]));

    if (end.dimension != 1) {
      CkAbort("ConfigurableRRMap only supports dimension 1!");
    }
	
    for (int index=0; index<maxIndex; index++) {	
      CkArrayIndex1D idx(index);		
      
      int cyclic_block = index / loader.objs_per_block;
      int cyclic_local = index % loader.objs_per_block;
      int l = loader.locations[ cyclic_local ];
      int PE = (cyclic_block*loader.PE_per_block + l) % CkNumPes();

      DEBUG(("[%d] ConfigurableRRMap: index=%d is located on PE %d l=%d\n", CkMyPe(), (int)index, (int)PE, l));

      if(PE == thisPe)
	mgr->insertInitial(idx,CkCopyMsg(&ctorMsg));

    }
    //        CKARRAYMAP_POPULATE_INITIAL(PE == thisPe);
	
    mgr->doneInserting();
    CkFreeMsg(ctorMsg);
  }
};


CkpvStaticDeclare(double*, rem);

class arrInfo {
 private:
   CkArrayIndex _nelems;
   int *_map;
 public:
   arrInfo(void):_map(NULL){}
   arrInfo(const CkArrayIndex& n, int *speeds)
   {
     _nelems = n;
     _map = new int[_nelems.getCombinedCount()];
     distrib(speeds);
   }
   ~arrInfo() { delete[] _map; }
   int getMap(const CkArrayIndex &i);
   void distrib(int *speeds);
   void pup(PUP::er& p){
     p|_nelems;
     int totalElements = _nelems.getCombinedCount();
     if(p.isUnpacking()){
       _map = new int[totalElements];
     }
     p(_map,totalElements);
   }
};

static int cmp(const void *first, const void *second)
{
  int fi = *((const int *)first);
  int si = *((const int *)second);
  return ((CkpvAccess(rem)[fi]==CkpvAccess(rem)[si]) ?
          0 :
          ((CkpvAccess(rem)[fi]<CkpvAccess(rem)[si]) ?
          1 : (-1)));
}

void
arrInfo::distrib(int *speeds)
{
  int _nelemsCount = _nelems.getCombinedCount();
  double total = 0.0;
  int npes = CkNumPes();
  int i,j,k;
  for(i=0;i<npes;i++)
    total += (double) speeds[i];
  double *nspeeds = new double[npes];
  for(i=0;i<npes;i++)
    nspeeds[i] = (double) speeds[i] / total;
  int *cp = new int[npes];
  for(i=0;i<npes;i++)
    cp[i] = (int) (nspeeds[i]*_nelemsCount);
  int nr = 0;
  for(i=0;i<npes;i++)
    nr += cp[i];
  nr = _nelemsCount - nr;
  if(nr != 0)
  {
    CkpvAccess(rem) = new double[npes];
    for(i=0;i<npes;i++)
      CkpvAccess(rem)[i] = (double)_nelemsCount*nspeeds[i] - cp[i];
    int *pes = new int[npes];
    for(i=0;i<npes;i++)
      pes[i] = i;
    qsort(pes, npes, sizeof(int), cmp);
    for(i=0;i<nr;i++)
      cp[pes[i]]++;
    delete[] pes;
    delete[] CkpvAccess(rem);
  }
  k = 0;
  for(i=0;i<npes;i++)
  {
    for(j=0;j<cp[i];j++)
      _map[k++] = i;
  }
  delete[] cp;
  delete[] nspeeds;
}

int
arrInfo::getMap(const CkArrayIndex &i)
{
  if(i.dimension == 1)
    return _map[i.data()[0]];
  else
    return _map[((i.hash()+739)%1280107)%_nelems.getCombinedCount()];
}

//Speeds maps processor number to "speed" (some sort of iterations per second counter)
// It is initialized by processor 0.
static int* speeds;

#if CMK_USE_PROP_MAP
typedef struct _speedmsg
{
  char hdr[CmiMsgHeaderSizeBytes];
  int node;
  int speed;
} speedMsg;

static void _speedHdlr(void *m)
{
  speedMsg *msg=(speedMsg *)m;
  if (CmiMyRank()==0)
    for (int pe=0;pe<CmiNodeSize(msg->node);pe++)
      speeds[CmiNodeFirst(msg->node)+pe] = msg->speed;  
  CmiFree(m);
}

// initnode call
void _propMapInit(void)
{
  speeds = new int[CkNumPes()];
  int hdlr = CkRegisterHandler(_speedHdlr);
  CmiPrintf("[%d]Measuring processor speed for prop. mapping...\n", CkMyPe());
  int s = LDProcessorSpeed();
  speedMsg msg;
  CmiSetHandler(&msg, hdlr);
  msg.node = CkMyNode();
  msg.speed = s;
  CmiSyncBroadcastAllAndFree(sizeof(msg), &msg);
  for(int i=0;i<CkNumNodes();i++)
    CmiDeliverSpecificMsg(hdlr);
}
#else
void _propMapInit(void)
{
  speeds = new int[CkNumPes()];
  int i;
  for(i=0;i<CkNumPes();i++)
    speeds[i] = 1;
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
  PropMap(CkMigrateMessage *m) {}
  int registerArray(const CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx = arrs.size();
    arrs.resize(idx+1);
    arrs[idx] = new arrInfo(numElements, speeds);
    return idx;
  }
  void unregisterArray(int idx)
  {
    arrs[idx].destroy();
  }
  int procNum(int arrayHdl, const CkArrayIndex &i)
  {
    return arrs[arrayHdl]->getMap(i);
  }
  void pup(PUP::er& p){
    int oldNumPes = -1;
    if(p.isPacking()){
      oldNumPes = CkNumPes();
    }
    p|oldNumPes;
    p|arrs;
    if(p.isUnpacking() && oldNumPes != CkNumPes()){
      for(int idx = 0; idx < arrs.length(); ++idx){
        arrs[idx]->distrib(speeds);
      }
    }
  }
};

class CkMapsInit : public Chare
{
public:
	CkMapsInit(CkArgMsg *msg) {
		//_defaultArrayMapID = CProxy_HilbertArrayMap::ckNew();
		_defaultArrayMapID = CProxy_DefaultArrayMap::ckNew();
		_fastArrayMapID = CProxy_FastArrayMap::ckNew();
		delete msg;
	}

	CkMapsInit(CkMigrateMessage *m) {}
};

// given an envelope of a Charm msg, find the recipient object pointer
CkMigratable * CkArrayMessageObjectPtr(envelope *env) {
  if (env->getMsgtype() != ForArrayEltMsg)
      return NULL;   // not an array msg

  ///@todo: Delegate this to the array manager which can then deal with ForArrayEltMsg
  CkArray *mgr = CProxy_CkArray(env->getArrayMgr()).ckLocalBranch();
  return mgr ? mgr->lookup(ck::ObjID(env->getRecipientID()).getElementID()) : NULL;
}

/****************************** Out-of-Core support ********************/

#if CMK_OUT_OF_CORE
CooPrefetchManager CkArrayElementPrefetcher;
CkpvDeclare(int,CkSaveRestorePrefetch);

/**
 * Return the out-of-core objid (from CooRegisterObject)
 * that this Converse message will access.  If the message
 * will not access an object, return -1.
 */
int CkArrayPrefetch_msg2ObjId(void *msg) {
  envelope *env=(envelope *)msg;
  CkMigratable *elt = CkArrayMessageObjectPtr(env);
  return elt?elt->prefetchObjID:-1;
}

/**
 * Write this object (registered with RegisterObject)
 * to this writable file.
 */
void CkArrayPrefetch_writeToSwap(FILE *swapfile,void *objptr) {
  CkMigratable *elt=(CkMigratable *)objptr;

  //Save the element's data to disk:
  PUP::toDisk p(swapfile);
  elt->virtual_pup(p);

  //Call the element's destructor in-place (so pointer doesn't change)
  CkpvAccess(CkSaveRestorePrefetch)=1;
  elt->~CkMigratable(); //< because destructor is virtual, destroys user class too.
  CkpvAccess(CkSaveRestorePrefetch)=0;
}
	
/**
 * Read this object (registered with RegisterObject)
 * from this readable file.
 */
void CkArrayPrefetch_readFromSwap(FILE *swapfile,void *objptr) {
  CkMigratable *elt=(CkMigratable *)objptr;
  //Call the element's migration constructor in-place
  CkpvAccess(CkSaveRestorePrefetch)=1;
  int ctorIdx=_chareTable[elt->thisChareType]->migCtor;
  elt->myRec->invokeEntry(elt,(CkMigrateMessage *)0,ctorIdx,true);
  CkpvAccess(CkSaveRestorePrefetch)=0;
  
  //Restore the element's data from disk:
  PUP::fromDisk p(swapfile);
  elt->virtual_pup(p);
}

static void _CkMigratable_prefetchInit(void) 
{
  CkpvExtern(int,CkSaveRestorePrefetch);
  CkpvAccess(CkSaveRestorePrefetch)=0;
  CkArrayElementPrefetcher.msg2ObjId=CkArrayPrefetch_msg2ObjId;
  CkArrayElementPrefetcher.writeToSwap=CkArrayPrefetch_writeToSwap;
  CkArrayElementPrefetcher.readFromSwap=CkArrayPrefetch_readFromSwap;
  CooRegisterManager(&CkArrayElementPrefetcher, _charmHandlerIdx);
}
#endif

/****************************** CkMigratable ***************************/
/**
 * This tiny class is used to convey information to the 
 * newly created CkMigratable object when its constructor is called.
 */
class CkMigratable_initInfo {
public:
	CkLocRec *locRec;
	int chareType;
	bool forPrefetch; /* If true, this creation is only a prefetch restore-from-disk.*/
};

CkpvStaticDeclare(CkMigratable_initInfo,mig_initInfo);


void _CkMigratable_initInfoInit(void) {
  CkpvInitialize(CkMigratable_initInfo,mig_initInfo);
#if CMK_OUT_OF_CORE
  _CkMigratable_prefetchInit();
#endif
}

void CkMigratable::commonInit(void) {
	CkMigratable_initInfo &i=CkpvAccess(mig_initInfo);
#if CMK_OUT_OF_CORE
	isInCore=true;
	if (CkpvAccess(CkSaveRestorePrefetch))
		return; /* Just restoring from disk--don't touch object */
	prefetchObjID=-1; //Unregistered
#endif
	myRec=i.locRec;
	thisIndexMax=myRec->getIndex();
	thisChareType=i.chareType;
	usesAtSync=false;
	usesAutoMeasure=true;
	barrierRegistered=false;

  local_state = OFF;
  prev_load = 0.0;
  can_reset = false;

#if CMK_LBDB_ON
  if (_lb_args.metaLbOn()) {
    atsync_iteration = myRec->getMetaBalancer()->get_iteration();
    myRec->getMetaBalancer()->AdjustCountForNewContributor(atsync_iteration);
  }
#endif

	/*
	FAULT_EVAC
	*/
	AsyncEvacuate(true);
}

CkMigratable::CkMigratable(void) {
	DEBC((AA "In CkMigratable constructor\n" AB));
	commonInit();
}
CkMigratable::CkMigratable(CkMigrateMessage *m): Chare(m) {
	commonInit();
}

int CkMigratable::ckGetChareType(void) const {return thisChareType;}

void CkMigratable::pup(PUP::er &p) {
	DEBM((AA "In CkMigratable::pup %s\n" AB,idx2str(thisIndexMax)));
	Chare::pup(p);
	p|thisIndexMax;
	p(usesAtSync);
  p(can_reset);
	p(usesAutoMeasure);
#if CMK_LBDB_ON 
	int readyMigrate = 0;
	if (p.isPacking()) readyMigrate = myRec->isReadyMigrate();
	p|readyMigrate;
	if (p.isUnpacking()) myRec->ReadyMigrate(readyMigrate);
#endif
	if(p.isUnpacking()) barrierRegistered=false;
	/*
		FAULT_EVAC
	*/
	p | asyncEvacuate;
	if(p.isUnpacking()){myRec->AsyncEvacuate(asyncEvacuate);}
	
	ckFinishConstruction();
}

void CkMigratable::ckDestroy(void) {
	DEBC((AA "In CkMigratable::ckDestroy %s\n" AB,idx2str(thisIndexMax)));
	myRec->destroy();
}

void CkMigratable::ckAboutToMigrate(void) { }
void CkMigratable::ckJustMigrated(void) { }
void CkMigratable::ckJustRestored(void) { }

CkMigratable::~CkMigratable() {
	DEBC((AA "In CkMigratable::~CkMigratable %s\n" AB,idx2str(thisIndexMax)));
#if CMK_OUT_OF_CORE
	isInCore=false;
	if (CkpvAccess(CkSaveRestorePrefetch)) 
		return; /* Just saving to disk--don't deregister anything. */
	/* We're really leaving or dying-- unregister from the ooc system*/
	if (prefetchObjID!=-1) {
		CooDeregisterObject(prefetchObjID);
		prefetchObjID=-1;
	}
#endif
	/*Might want to tell myRec about our doom here--
	it's difficult to avoid some kind of circular-delete, though.
	*/
#if CMK_LBDB_ON 
	if (barrierRegistered) {
	  DEBL((AA "Removing barrier for element %s\n" AB,idx2str(thisIndexMax)));
	  if (usesAtSync)
		myRec->getLBDB()->RemoveLocalBarrierClient(ldBarrierHandle);
	  else
		myRec->getLBDB()->RemoveLocalBarrierReceiver(ldBarrierRecvHandle);
	}

  if (_lb_args.metaLbOn()) {
    myRec->getMetaBalancer()->AdjustCountForDeadContributor(atsync_iteration);
  }
#endif
	//To detect use-after-delete
	thisIndexMax.nInts=-12345;
	thisIndexMax.dimension=-12345;
}

void CkMigratable::CkAbort(const char *why) const {
	CkError("CkMigratable '%s' aborting:\n",_chareTable[thisChareType]->name);
	::CkAbort(why);
}

void CkMigratable::ResumeFromSync(void)
{
//	CkAbort("::ResumeFromSync() not defined for this array element!\n");
}

void CkMigratable::UserSetLBLoad() {
	CkAbort("::UserSetLBLoad() not defined for this array element!\n");
}

#if CMK_LBDB_ON  //For load balancing:
// user can call this helper function to set obj load (for model-based lb)
void CkMigratable::setObjTime(double cputime) {
	myRec->setObjTime(cputime);
}
double CkMigratable::getObjTime() {
	return myRec->getObjTime();
}

#if CMK_LB_USER_DATA
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
void *CkMigratable::getObjUserData(int idx) {
	return myRec->getObjUserData(idx);
}
#endif

void CkMigratable::clearMetaLBData() {
//  if (can_reset) {
    local_state = OFF;
    atsync_iteration = -1;
    prev_load = 0.0;
    can_reset = false;
//  }
}

void CkMigratable::recvLBPeriod(void *data) {
  if (atsync_iteration < 0) {
    return;
  }
  int lb_period = *((int *) data);
 DEBAD(("\t[obj %s] Received the LB Period %d current iter %d state %d on PE %d\n",
     idx2str(thisIndexMax), lb_period, atsync_iteration, local_state, CkMyPe()));

  bool is_tentative;
  if (local_state == LOAD_BALANCE) {
    CkAssert(lb_period == myRec->getMetaBalancer()->getPredictedLBPeriod(is_tentative));
    return;
  }

  if (local_state == PAUSE) {
    if (atsync_iteration < lb_period) {
      local_state = DECIDED;
      ResumeFromSync();
      return;
    }
    local_state = LOAD_BALANCE;

    can_reset = true;
    //myRec->getLBDB()->AtLocalBarrier(ldBarrierHandle);
    return;
  }
  local_state = DECIDED;
}

void CkMigratable::metaLBCallLB() {
	myRec->getLBDB()->AtLocalBarrier(ldBarrierHandle);
}

void CkMigratable::ckFinishConstruction(void)
{
//	if ((!usesAtSync) || barrierRegistered) return;
	myRec->setMeasure(usesAutoMeasure);
	if (barrierRegistered) return;
	DEBL((AA "Registering barrier client for %s\n" AB,idx2str(thisIndexMax)));
        if (usesAtSync)
	  ldBarrierHandle = myRec->getLBDB()->AddLocalBarrierClient(
		(LDBarrierFn)staticResumeFromSync,(void*)(this));
        else
	  ldBarrierRecvHandle = myRec->getLBDB()->AddLocalBarrierReceiver(
		(LDBarrierFn)staticResumeFromSync,(void*)(this));
	barrierRegistered=true;
}

void CkMigratable::AtSync(int waitForMigration)
{
	if (!usesAtSync)
		CkAbort("You must set usesAtSync=true in your array element constructor to use AtSync!\n");
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        mlogData->toResumeOrNot=1;
#endif
	myRec->AsyncMigrate(!waitForMigration);
	if (waitForMigration) ReadyMigrate(true);
	ckFinishConstruction();
  DEBL((AA "Element %s going to sync\n" AB,idx2str(thisIndexMax)));
  // model-based load balancing, ask user to provide cpu load
  if (usesAutoMeasure == false) UserSetLBLoad();

  PUP::sizer ps;
  this->virtual_pup(ps);
  setPupSize(ps.size());

  if (!_lb_args.metaLbOn()) {
    myRec->getLBDB()->AtLocalBarrier(ldBarrierHandle);
    return;
  }

  // When MetaBalancer is turned on

  if (atsync_iteration == -1) {
    can_reset = false;
    local_state = OFF;
    prev_load = 0.0;
  }

  atsync_iteration++;
  //CkPrintf("[pe %s] atsync_iter %d && predicted period %d state: %d\n",
  //    idx2str(thisIndexMax), atsync_iteration,
  //    myRec->getMetaBalancer()->getPredictedLBPeriod(), local_state);
  double tmp = prev_load;
  prev_load = myRec->getObjTime();
  double current_load = prev_load - tmp;

  // If the load for the chares are based on certain model, then set the
  // current_load to be whatever is the obj load.
  if (!usesAutoMeasure) {
    current_load = myRec->getObjTime();
  }

  if (atsync_iteration <= myRec->getMetaBalancer()->get_finished_iteration()) {
    CkPrintf("[%d:%s] Error!! Contributing to iter %d < current iter %d\n",
      CkMyPe(), idx2str(thisIndexMax), atsync_iteration,
      myRec->getMetaBalancer()->get_finished_iteration());
    CkAbort("Not contributing to the right iteration\n");
  }

  if (atsync_iteration != 0) {
    myRec->getMetaBalancer()->AddLoad(atsync_iteration, current_load);
  }

  bool is_tentative;
  if (atsync_iteration < myRec->getMetaBalancer()->getPredictedLBPeriod(is_tentative)) {
    ResumeFromSync();
  } else if (is_tentative) {
    local_state = PAUSE;
  } else if (local_state == DECIDED) {
    DEBAD(("[%d:%s] Went to load balance iter %d\n", CkMyPe(), idx2str(thisIndexMax), atsync_iteration));
    local_state = LOAD_BALANCE;
    can_reset = true;
    //myRec->getLBDB()->AtLocalBarrier(ldBarrierHandle);
  } else {
    DEBAD(("[%d:%s] Went to pause state iter %d\n", CkMyPe(), idx2str(thisIndexMax), atsync_iteration));
    local_state = PAUSE;
  }
}

void CkMigratable::ReadyMigrate(bool ready)
{
	myRec->ReadyMigrate(ready);
}

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    extern int globalResumeCount;
#endif

void CkMigratable::staticResumeFromSync(void* data)
{
	CkMigratable *el=(CkMigratable *)data;
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    if(el->mlogData->toResumeOrNot ==0 || el->mlogData->resumeCount >= globalResumeCount){
        return;
    }
#endif
	DEBL((AA "Element %s resuming from sync\n" AB,idx2str(el->thisIndexMax)));
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CpvAccess(_currentObj) = el;
#endif

  if (_lb_args.metaLbOn()) {
  	el->clearMetaLBData();
	}
	el->ResumeFromSync();
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    el->mlogData->resumeCount++;
#endif
}

void CkMigratable::setMigratable(int migratable) 
{
	myRec->setMigratable(migratable);
}

void CkMigratable::setPupSize(size_t obj_pup_size)
{
	myRec->setPupSize(obj_pup_size);
}

struct CkArrayThreadListener {
        struct CthThreadListener base;
        CkMigratable *mig;
};

extern "C"
void CkArrayThreadListener_suspend(struct CthThreadListener *l)
{
        CkArrayThreadListener *a=(CkArrayThreadListener *)l;
        a->mig->ckStopTiming();
}

extern "C"
void CkArrayThreadListener_resume(struct CthThreadListener *l)
{
        CkArrayThreadListener *a=(CkArrayThreadListener *)l;
        a->mig->ckStartTiming();
}

extern "C"
void CkArrayThreadListener_free(struct CthThreadListener *l)
{
        CkArrayThreadListener *a=(CkArrayThreadListener *)l;
        delete a;
}

void CkMigratable::CkAddThreadListeners(CthThread tid, void *msg)
{
        Chare::CkAddThreadListeners(tid, msg);   // for trace
        CthSetThreadID(tid, thisIndexMax.data()[0], thisIndexMax.data()[1], 
		       thisIndexMax.data()[2]);
	CkArrayThreadListener *a=new CkArrayThreadListener;
	a->base.suspend=CkArrayThreadListener_suspend;
	a->base.resume=CkArrayThreadListener_resume;
	a->base.free=CkArrayThreadListener_free;
	a->mig=this;
	CthAddListener(tid,(struct CthThreadListener *)a);
}
#else
void CkMigratable::setObjTime(double cputime) {}
double CkMigratable::getObjTime() {return 0.0;}

#if CMK_LB_USER_DATA
void *CkMigratable::getObjUserData(int idx) { return NULL; }
#endif

/* no load balancer: need dummy implementations to prevent link error */
void CkMigratable::CkAddThreadListeners(CthThread tid, void *msg)
{
}
#endif


/************************** Location Records: *********************************/


/*----------------- Local:
Matches up the array index with the local index, an
interfaces with the load balancer on behalf of the
represented array elements.
*/
CkLocRec::CkLocRec(CkLocMgr *mgr,bool fromMigration,
                   bool ignoreArrival, const CkArrayIndex &idx_, CmiUInt8 id_)
  :myLocMgr(mgr),idx(idx_), id(id_),
	 running(false),deletedMarker(NULL)
{
#if CMK_LBDB_ON
	DEBL((AA "Registering element %s with load balancer\n" AB,idx2str(idx)));
	//BIGSIM_OOC DEBUGGING
	//CkPrintf("LocMgr on %d: Registering element %s with load balancer\n", CkMyPe(), idx2str(idx));
	nextPe = -1;
	asyncMigrate = false;
	readyMigrate = true;
        enable_measure = true;
	bounced  = false;
	the_lbdb=mgr->getLBDB();
	the_metalb=mgr->getMetaBalancer();
	LDObjid ldid = idx2LDObjid(idx);
#if CMK_GLOBAL_LOCATION_UPDATE
        ldid.locMgrGid = mgr->getGroupID().idx;
#endif        
	ldHandle=the_lbdb->RegisterObj(mgr->getOMHandle(),
		ldid,(void *)this,1);
	if (fromMigration) {
		DEBL((AA "Element %s migrated in\n" AB,idx2str(idx)));
		if (!ignoreArrival)  {
			the_lbdb->Migrated(ldHandle, true);
		  // load balancer should ignore this objects movement
		//  AsyncMigrate(true);
		}
	}
#endif
	/*
		FAULT_EVAC
	*/
	asyncEvacuate = true;
}
CkLocRec::~CkLocRec()
{
	if (deletedMarker!=NULL) *deletedMarker=true;
	myLocMgr->reclaim(idx);
#if CMK_LBDB_ON
	stopTiming();
	DEBL((AA "Unregistering element %s from load balancer\n" AB,idx2str(idx)));
	the_lbdb->UnregisterObj(ldHandle);
#endif
}
void CkLocRec::migrateMe(int toPe) //Leaving this processor
{
	//This will pack us up, send us off, and delete us
//	printf("[%d] migrating migrateMe to %d \n",CkMyPe(),toPe);
	myLocMgr->emigrate(this,toPe);
}

#if CMK_LBDB_ON
void CkLocRec::startTiming(int ignore_running) {
  	if (!ignore_running) running=true;
	DEBL((AA "Start timing for %s at %.3fs {\n" AB,idx2str(idx),CkWallTimer()));
  	if (enable_measure) the_lbdb->ObjectStart(ldHandle);
}
void CkLocRec::stopTiming(int ignore_running) {
	DEBL((AA "} Stop timing for %s at %.3fs\n" AB,idx2str(idx),CkWallTimer()));
  	if ((ignore_running || running) && enable_measure) the_lbdb->ObjectStop(ldHandle);
  	if (!ignore_running) running=false;
}
void CkLocRec::setObjTime(double cputime) {
	the_lbdb->EstObjLoad(ldHandle, cputime);
}
double CkLocRec::getObjTime() {
        LBRealType walltime, cputime;
        the_lbdb->GetObjLoad(ldHandle, walltime, cputime);
        return walltime;
}
#if CMK_LB_USER_DATA
void* CkLocRec::getObjUserData(int idx) {
        return the_lbdb->GetDBObjUserData(ldHandle, idx);
}
#endif
#endif

void CkLocRec::destroy(void) //User called destructor
{
	//Our destructor does all the needed work
	delete this;
}

/**********Added for cosmology (inline function handling without parameter marshalling)***********/

LDObjHandle CkMigratable::timingBeforeCall(int* objstopped){
	LDObjHandle objHandle;
#if CMK_LBDB_ON
	if (getLBDB()->RunningObject(&objHandle)) {
		*objstopped = 1;
		getLBDB()->ObjectStop(objHandle);
	}
	myRec->startTiming(1);
#endif

  return objHandle;
}

void CkMigratable::timingAfterCall(LDObjHandle objHandle,int *objstopped){
	myRec->stopTiming(1);
#if CMK_LBDB_ON
	if (*objstopped) {
		 getLBDB()->ObjectStart(objHandle);
	}
#endif

 return;
}
/****************************************************************************/


bool CkLocRec::invokeEntry(CkMigratable *obj,void *msg,
	int epIdx,bool doFree) 
{

	DEBS((AA "   Invoking entry %d on element %s\n" AB,epIdx,idx2str(idx)));
	bool isDeleted=false; //Enables us to detect deletion during processing
	deletedMarker=&isDeleted;
	startTiming();


#if CMK_TRACE_ENABLED
	if (msg) { /* Tracing: */
		envelope *env=UsrToEnv(msg);
	//	CkPrintf("ckLocation.C beginExecuteDetailed %d %d \n",env->getEvent(),env->getsetArraySrcPe());
		if (_entryTable[epIdx]->traceEnabled)
        {
            _TRACE_BEGIN_EXECUTE_DETAILED(env->getEvent(), ForChareMsg,epIdx,env->getSrcPe(), env->getTotalsize(), idx.getProjectionID(env->getArrayMgrIdx()), obj);
            if(_entryTable[epIdx]->appWork)
                _TRACE_BEGIN_APPWORK();
        }
	}
#endif

	if (doFree) 
	   CkDeliverMessageFree(epIdx,msg,obj);
	else /* !doFree */
	   CkDeliverMessageReadonly(epIdx,msg,obj);


#if CMK_TRACE_ENABLED
	if (msg) { /* Tracing: */
		if (_entryTable[epIdx]->traceEnabled)
        {
            if(_entryTable[epIdx]->appWork)
                _TRACE_END_APPWORK();
			_TRACE_END_EXECUTE();
        }
	}
#endif
#if CMK_LBDB_ON
        if (!isDeleted) checkBufferedMigration();   // check if should migrate
#endif
	if (isDeleted) return false;//We were deleted
	deletedMarker=NULL;
	stopTiming();
	return true;
}

#if CMK_LBDB_ON

void CkLocRec::staticMetaLBResumeWaitingChares(LDObjHandle h, int lb_ideal_period) {
	CkLocRec *el=(CkLocRec *)LDObjUserData(h);
	DEBL((AA "MetaBalancer wants to resume waiting chare %s\n" AB,idx2str(el->idx)));
	el->myLocMgr->informLBPeriod(el, lb_ideal_period);
}

void CkLocRec::staticMetaLBCallLBOnChares(LDObjHandle h) {
	CkLocRec *el=(CkLocRec *)LDObjUserData(h);
	DEBL((AA "MetaBalancer wants to call LoadBalance on chare %s\n" AB,idx2str(el->idx)));
	el->myLocMgr->metaLBCallLB(el);
}

void CkLocRec::staticMigrate(LDObjHandle h, int dest)
{
	CkLocRec *el=(CkLocRec *)LDObjUserData(h);
	DEBL((AA "Load balancer wants to migrate %s to %d\n" AB,idx2str(el->idx),dest));
	el->recvMigrate(dest);
}

void CkLocRec::recvMigrate(int toPe)
{
	// we are in the mode of delaying actual migration
 	// till readyMigrate()
	if (readyMigrate) { migrateMe(toPe); }
	else nextPe = toPe;
}

void CkLocRec::AsyncMigrate(bool use)  
{
        asyncMigrate = use; 
	the_lbdb->UseAsyncMigrate(ldHandle, use);
}

bool CkLocRec::checkBufferedMigration()
{
	// we don't migrate in user's code when calling ReadyMigrate(true)
	// we postphone the action to here until we exit from the user code.
	if (readyMigrate && nextPe != -1) {
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
  	  the_lbdb->Migratable(ldHandle);
	else
  	  the_lbdb->NonMigratable(ldHandle);
}

void CkLocRec::setPupSize(size_t obj_pup_size) {
  the_lbdb->setPupSize(ldHandle, obj_pup_size);
}

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void CkLocRec::Migrated(){
    the_lbdb->Migrated(ldHandle, true);
}
#endif
#endif


//doesn't delete if there is extra pe
void CkLocMgr::flushLocalRecs(void)
{
  void *objp;
  void *keyp;
  CkHashtableIterator *it=hash.iterator();
  CmiImmediateLock(hashImmLock);
  while (NULL!=(objp=it->next(&keyp))) {
    CkLocRec *rec=*(CkLocRec **)objp;
    CkArrayIndex &idx=*(CkArrayIndex *)keyp;
    callMethod((CkLocRec*)rec, &CkMigratable::ckDestroy);
    it->seek(-1);//retry this hash slot
  }
  delete it;
  CmiImmediateUnlock(hashImmLock);
}

// clean all buffer'ed messages and also free local objects
void CkLocMgr::flushAllRecs(void)
{
  void *objp;
  void *keyp;
  CkHashtableIterator *it=hash.iterator();
  CmiImmediateLock(hashImmLock);
  while (NULL!=(objp=it->next(&keyp))) {
    CkLocRec *rec=*(CkLocRec **)objp;
    callMethod(rec, &CkMigratable::ckDestroy);
    it->seek(-1);//retry this hash slot
  }
  delete it;
  CmiImmediateUnlock(hashImmLock);
}


#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void CkLocMgr::callForAllRecords(CkLocFn fnPointer,CkArray *arr,void *data){
	void *objp;
	void *keyp;

	CkHashtableIterator *it = hash.iterator();
	while (NULL!=(objp=it->next(&keyp))) {
		CkLocRec *rec=*(CkLocRec **)objp;
		CkArrayIndex &idx=*(CkArrayIndex *)keyp;
		fnPointer(arr,data,rec,&idx);
	}

	// releasing iterator memory
	delete it;
}
#endif

/*************************** LocMgr: CREATION *****************************/
CkLocMgr::CkLocMgr(CkArrayOptions opts)
	:thisProxy(thisgroup),thislocalproxy(thisgroup,CkMyPe()),
	 hash(17,0.3)
	, idCounter(1)
        , bounds(opts.getBounds())
{
	DEBC((AA "Creating new location manager %d\n" AB,thisgroup));
// moved to _CkMigratable_initInfoInit()
//	CkpvInitialize(CkMigratable_initInfo,mig_initInfo);

	duringMigration = false;

//Register with the map object
	mapID = opts.getMap();
	map=(CkArrayMap *)CkLocalBranch(mapID);
	if (map==NULL) CkAbort("ERROR!  Local branch of array map is NULL!");
	mapHandle=map->registerArray(opts.getEnd(), thisgroup);

        // Figure out the mapping from indices to object IDs if one is possible
        compressor = ck::FixedArrayIndexCompressor::make(bounds);

//Find and register with the load balancer
#if CMK_LBDB_ON
        lbdbID = _lbdb;
        metalbID = _metalb;
#endif
        initLB(lbdbID, metalbID);
	hashImmLock = CmiCreateImmediateLock();
}

CkLocMgr::CkLocMgr(CkMigrateMessage* m)
	:IrrGroup(m),thisProxy(thisgroup),thislocalproxy(thisgroup,CkMyPe()),hash(17,0.3)
{
	duringMigration = false;
	hashImmLock = CmiCreateImmediateLock();
}

CkLocMgr::~CkLocMgr() {
#if CMK_LBDB_ON
  the_lbdb->RemoveLocalBarrierClient(dummyBarrierHandle);
  the_lbdb->DecreaseLocalBarrier(dummyBarrierHandle, 1);
  the_lbdb->RemoveLocalBarrierReceiver(lbBarrierReceiver);
  the_lbdb->UnregisterOM(myLBHandle);
#endif
  map->unregisterArray(mapHandle);
  CmiDestroyLock(hashImmLock);
}

void CkLocMgr::pup(PUP::er &p){
	IrrGroup::pup(p);
	p|mapID;
	p|mapHandle;
	p|lbdbID;
        p|metalbID;
        p|bounds;
	if(p.isUnpacking()) {
		thisProxy=thisgroup;
		CProxyElement_CkLocMgr newlocalproxy(thisgroup,CkMyPe());
		thislocalproxy=newlocalproxy;
		//Register with the map object
		map=(CkArrayMap *)CkLocalBranch(mapID);
		if (map==NULL) CkAbort("ERROR!  Local branch of array map is NULL!");
                CkArrayIndex emptyIndex;
		// _lbdb is the fixed global groupID
		initLB(lbdbID, metalbID);
                compressor = ck::FixedArrayIndexCompressor::make(bounds);
#if __FAULT__
        int count = 0;
        p | count;
        DEBUG(CmiPrintf("[%d] Unpacking Locmgr %d has %d home elements\n",CmiMyPe(),thisgroup.idx,count));
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))    
        homeElementCount = count;
#endif
        for(int i=0;i<count;i++){
            CkArrayIndex idx;
            int pe = 0;
            idx.pup(p);
            p | pe;
  //          CmiPrintf("[%d] idx %s is a home element exisiting on pe %d\n",CmiMyPe(),idx2str(idx),pe);
            inform(idx, idx2id[idx], pe);
            CmiUInt8 id = lookupID(idx);
            CkLocRec *rec = elementNrec(id);
            CmiAssert(rec!=NULL);
            CmiAssert(lastKnown(idx) == pe);
        }
#endif
		// delay doneInserting when it is unpacking during restart.
		// to prevent load balancing kicking in
		if (!CkInRestarting()) 
			doneInserting();
	}else{
 /**
 * pack the indexes of elements which have their homes on this processor
 * but dont exist on it.. needed for broadcast after a restart
 * indexes of local elements dont need to be packed
 * since they will be recreated later anyway
 */
#if __FAULT__
        int count=0;
        CkVec<int> pe_list;
        CkVec<CmiUInt8> idx_list;
        for (std::map<CmiUInt8, int>::iterator itr = id2pe.begin(); itr != id2pe.end(); ++itr)
            if (homePe(itr->first) == CmiMyPe() && itr->second != CmiMyPe())
            {
                idx_list.push_back(itr->first);
                pe_list.push_back(itr->second);
                count++;
            }

        p | count;
      for(int i=0;i<pe_list.length();i++){
        p|idx_list[i];
        p|pe_list[i];
      }
#endif

	}
}

/// Add a new local array manager to our list.
void CkLocMgr::addManager(CkArrayID id,CkArray *mgr)
{
	CK_MAGICNUMBER_CHECK
	DEBC((AA "Adding new array manager\n" AB));
	managers[id] = mgr;
}

void CkLocMgr::deleteManager(CkArrayID id, CkArray *mgr) {
  CkAssert(managers[id] == mgr);
  managers.erase(id);

  if (managers.size() == 0)
    delete this;
}

//Tell this element's home processor it now lives "there"
void CkLocMgr::informHome(const CkArrayIndex &idx,int nowOnPe)
{
	int home=homePe(idx);
	if (home!=CkMyPe() && home!=nowOnPe) {
		//Let this element's home Pe know it lives here now
		DEBC((AA "  Telling %s's home %d that it lives on %d.\n" AB,idx2str(idx),home,nowOnPe));
//#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
//#if defined(_FAULT_MLOG_)
//        informLocationHome(thisgroup,idx,home,CkMyPe());
//#else
		thisProxy[home].updateLocation(idx, lookupID(idx), nowOnPe);
//#endif
	}
}

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
CkLocRec *CkLocMgr::createLocal(const CkArrayIndex &idx,
        bool forMigration, bool ignoreArrival,
        bool notifyHome,int dummy)
{
    DEBC((AA "Adding new record for element %s\n" AB,idx2str(idx)));
    CkLocRec *rec=new CkLocRec(this,forMigration,ignoreArrival,idx);
    if(!dummy){
        insertRec(rec,idx); //Add to global hashtable
        id2pe[idx] = CkMyPe();
        deliverAnyBufferedMsgs(idx, bufferedMsgs);
    }   
    if (notifyHome) informHome(idx,CkMyPe());
    return rec; 
}
#else
CkLocRec *CkLocMgr::createLocal(const CkArrayIndex &idx, 
		bool forMigration, bool ignoreArrival,
		bool notifyHome)
{
	DEBC((AA "Adding new record for element %s\n" AB,idx2str(idx)));
	CmiUInt8 id = lookupID(idx);

	CkLocRec *rec=new CkLocRec(this, forMigration, ignoreArrival, idx, id);
	insertRec(rec, id);
        inform(idx, id, CkMyPe());

	if (notifyHome) { informHome(idx,CkMyPe()); }
	return rec;
}
#endif


void CkLocMgr::deliverAnyBufferedMsgs(CmiUInt8 id, MsgBuffer &buffer)
{
    MsgBuffer::iterator itr = buffer.find(id);
    // If there are no buffered msgs, don't do anything
    if (itr == buffer.end()) return;

    std::vector<CkArrayMessage*> messagesToFlush;
    messagesToFlush.swap(itr->second);

    // deliver all buffered messages
    for (int i = 0; i < messagesToFlush.size(); ++i)
    {
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        envelope *env = UsrToEnv(messagesToFlush[i]);
        Chare *oldObj = CpvAccess(_currentObj);
        CpvAccess(_currentObj) =(Chare *) env->sender.getObject();
        env->sender.type = TypeInvalid;
#endif
        CkArrayMessage *m = messagesToFlush[i];
        deliverMsg(m, UsrToEnv(m)->getArrayMgr(), id, NULL, CkDeliver_queue);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        CpvAccess(_currentObj) = oldObj;
#endif
    }

    CkAssert(itr->second.empty()); // Nothing should have been added, since we
                                   // ostensibly know where the object lives

    // and then, delete the entry in this table of buffered msgs
    buffer.erase(itr);
}

CmiUInt8 CkLocMgr::getNewObjectID(const CkArrayIndex &idx)
{
  CmiUInt8 id;

  if (compressor) {
    id = compressor->compress(idx);
    idx2id[idx] = id;
    return id;
  }

  CkLocMgr::IdxIdMap::iterator itr = idx2id.find(idx);
  if (itr == idx2id.end()) {
    id = idCounter++ + ((CmiUInt8)CkMyPe() << 24);
    idx2id[idx] = id;
  } else {
    id = itr->second;
  }
  return id;
}

//Add a new local array element, calling element's constructor
bool CkLocMgr::addElement(CkArrayID mgr,const CkArrayIndex &idx,
		CkMigratable *elt,int ctorIdx,void *ctorMsg)
{
	CK_MAGICNUMBER_CHECK

        CmiUInt8 id = getNewObjectID(idx);

	CkLocRec *rec = elementNrec(id);
	if (rec == NULL)
	{ //This is the first we've heard of that element-- add new local record
		rec=createLocal(idx,false,false,true);
#if CMK_GLOBAL_LOCATION_UPDATE
                if (homePe(idx) != CkMyPe()) {
                  DEBC((AA "Global location broadcast for new element idx %s "
                        "assigned to %d \n" AB, idx2str(idx), CkMyPe()));
                  thisProxy.updateLocation(idx, CkMyPe());  
                }
#endif
                
	}
	//rec is *already* local-- must not be the first insertion	
        else
		deliverAnyBufferedMsgs(id, bufferedShadowElemMsgs);
	if (!addElementToRec(rec, managers[mgr], elt, ctorIdx, ctorMsg)) return false;
	elt->ckFinishConstruction();
	return true;
}

//As above, but shared with the migration code
bool CkLocMgr::addElementToRec(CkLocRec *rec,CkArray *mgr,
		CkMigratable *elt,int ctorIdx,void *ctorMsg)
{//Insert the new element into its manager's local list
  CmiUInt8 id = lookupID(rec->getIndex());
  if (mgr->getEltFromArrMgr(id))
    CkAbort("Cannot insert array element twice!");
  mgr->putEltInArrMgr(id, elt); //Local element table

//Call the element's constructor
	DEBC((AA "Constructing element %s of array\n" AB,idx2str(rec->getIndex())));
	CkMigratable_initInfo &i=CkpvAccess(mig_initInfo);
	i.locRec=rec;
	i.chareType=_entryTable[ctorIdx]->chareIdx;

#ifndef CMK_CHARE_USE_PTR
  int callingChareIdx = CkpvAccess(currentChareIdx);
  CkpvAccess(currentChareIdx) = -1;
#endif

	if (!rec->invokeEntry(elt,ctorMsg,ctorIdx,true)) return false;

#ifndef CMK_CHARE_USE_PTR
  CkpvAccess(currentChareIdx) = callingChareIdx;
#endif

#if CMK_OUT_OF_CORE
	/* Register new element with out-of-core */
	PUP::sizer p_getSize;
	elt->virtual_pup(p_getSize);
	elt->prefetchObjID=CooRegisterObject(&CkArrayElementPrefetcher,p_getSize.size(),elt);
#endif
	
	return true;
}

void CkLocMgr::requestLocation(const CkArrayIndex &idx, const int peToTell,
                               bool suppressIfHere, int ifNonExistent, int chareType, CkArrayID mgr) {
  int onPe = -1;
  DEBN(("%d requestLocation for %s peToTell %d\n", CkMyPe(), idx2str(idx), peToTell));

  if (peToTell == CkMyPe())
    return;

  CkLocMgr::IdxIdMap::iterator itr = idx2id.find(idx);
  if (itr == idx2id.end()) {
    DEBN(("%d Buffering ID/location req for %s\n", CkMyPe(), idx2str(idx)));
    bufferedLocationRequests[idx].push_back(make_pair(peToTell, suppressIfHere));

    switch (ifNonExistent) {
    case CkArray_IfNotThere_createhome:
      demandCreateElement(idx, chareType, CkMyPe(), mgr);
      break;
    case CkArray_IfNotThere_createhere:
      demandCreateElement(idx, chareType, peToTell, mgr);
      break;
    default:
      break;
    }

    return;
  }

  onPe = lastKnown(idx);

  if (suppressIfHere && peToTell == CkMyPe())
    return;

  thisProxy[peToTell].updateLocation(idx, idx2id[idx], onPe);
}

void CkLocMgr::requestLocation(CmiUInt8 id, const int peToTell,
                               bool suppressIfHere) {
  int onPe = -1;
  DEBN(("%d requestLocation for %u peToTell %d\n", CkMyPe(), id, peToTell));

  if (peToTell == CkMyPe())
    return;

  onPe = lastKnown(id);

  if (suppressIfHere && peToTell == CkMyPe())
    return;

  thisProxy[peToTell].updateLocation(id, onPe);
}

void CkLocMgr::updateLocation(const CkArrayIndex &idx, CmiUInt8 id, int nowOnPe) {
  DEBN(("%d updateLocation for %s on %d\n", CkMyPe(), idx2str(idx), nowOnPe));
  inform(idx, id, nowOnPe);
  deliverAnyBufferedMsgs(id, bufferedRemoteMsgs);
}

void CkLocMgr::updateLocation(CmiUInt8 id, int nowOnPe) {
  DEBN(("%d updateLocation for %s on %d\n", CkMyPe(), idx2str(idx), nowOnPe));
  inform(id, nowOnPe);
  deliverAnyBufferedMsgs(id, bufferedRemoteMsgs);
}

void CkLocMgr::inform(const CkArrayIndex &idx, CmiUInt8 id, int nowOnPe) {
  // On restart, conservatively determine the next 'safe' ID to
  // generate for new elements by the max over all of the elements with
  // IDs corresponding to each PE
  if (CkInRestarting()) {
    CmiUInt8 maskedID = id & ((1u << 24) - 1);
    CmiUInt8 origPe = id >> 24;
    if (origPe == CkMyPe()) {
      if (maskedID >= idCounter)
        idCounter = maskedID + 1;
    } else {
      if (origPe < CkNumPes())
        thisProxy[origPe].updateLocation(idx, id, nowOnPe);
    }
  }

  idx2id[idx] = id;
  id2pe[id] = nowOnPe;

  std::map<CkArrayIndex, std::vector<std::pair<int, bool> > >::iterator itr =
    bufferedLocationRequests.find(idx);
  if (itr != bufferedLocationRequests.end()) {
    for (std::vector<std::pair<int, bool> >::iterator i = itr->second.begin();
         i != itr->second.end(); ++i) {
      int peToTell = i->first;
      DEBN(("%d Replying to buffered ID/location req to pe %d\n", CkMyPe(), peToTell));
      if (peToTell != CkMyPe())
        thisProxy[peToTell].updateLocation(idx, id, nowOnPe);
    }
    bufferedLocationRequests.erase(itr);
  }

  deliverAnyBufferedMsgs(id, bufferedMsgs);

  std::map<CkArrayIndex, std::vector<CkArrayMessage*> >::iterator idx_itr =
    bufferedIndexMsgs.find(idx);
  if (idx_itr != bufferedIndexMsgs.end()) {
    vector<CkArrayMessage*> &msgs = idx_itr->second;
    for (int i = 0; i < msgs.size(); ++i) {
      envelope *env = UsrToEnv(msgs[i]);
      CkGroupID mgr = ck::ObjID(env->getRecipientID()).getCollectionID();
      env->setRecipientID(ck::ObjID(mgr, id));
      deliverMsg(msgs[i], mgr, id, &idx, CkDeliver_queue);
    }
    bufferedIndexMsgs.erase(idx_itr);
  }
}

void CkLocMgr::inform(CmiUInt8 id, int nowOnPe) {
  id2pe[id] = nowOnPe;
  deliverAnyBufferedMsgs(id, bufferedMsgs);
}



/*************************** LocMgr: DELETION *****************************/
/// This index will no longer be used-- delete the associated elements
void CkLocMgr::reclaim(const CkArrayIndex &idx) {
	CK_MAGICNUMBER_CHECK
	DEBC((AA "Destroying element %s\n" AB,idx2str(idx)));
	//Delete, and mark as empty, each array element
        CmiUInt8 id = lookupID(idx);
    for (std::map<CkArrayID, CkArray*>::iterator itr = managers.begin();
            itr != managers.end(); ++itr) {
      itr->second->deleteElt(id);
    }
	
	removeFromTable(id);
	
	if (!duringMigration) 
	{ //This is a local element dying a natural death
#if CMK_BIGSIM_CHARM
		//After migration, reclaimRemote will be called through 
		//the CkRemoveArrayElement in the pupping routines for those 
		//objects that are not on the home processors. However,
		//those remote records should not be deleted since the corresponding
		//objects are not actually deleted but on disk. If deleted, msgs
		//that seeking where is the object will be accumulated (a circular
		//msg chain) and causes the program no progress
		if(_BgOutOfCoreFlag==1) return; 
#endif
		int home=homePe(idx);
		if (home!=CkMyPe())
#if CMK_MEM_CHECKPOINT
	        if (!CkInRestarting()) // all array elements are removed anyway
#endif
	        if (!duringDestruction)
	            thisProxy[home].reclaimRemote(idx,CkMyPe());
	}
}

void CkLocMgr::reclaimRemote(const CkArrayIndex &idx,int deletedOnPe) {
	DEBC((AA "Our element %s died on PE %d\n" AB,idx2str(idx),deletedOnPe));

        CmiUInt8 id;
        if (!lookupID(idx, id)) return;

	CkLocRec *rec=elementNrec(id);
	if (rec==NULL) return; //We never knew him
	removeFromTable(id);
	delete rec;
}
void CkLocMgr::removeFromTable(const CmiUInt8 id) {
#if CMK_ERROR_CHECKING
	//Make sure it's actually in the table before we delete it
	if (NULL==elementNrec(id))
		CkAbort("CkLocMgr::removeFromTable called on invalid index!");
#endif
        CmiImmediateLock(hashImmLock);
	hash.remove(id);
        CmiImmediateUnlock(hashImmLock);
#if CMK_ERROR_CHECKING
	//Make sure it's really gone
	if (NULL!=elementNrec(id))
		CkAbort("CkLocMgr::removeFromTable called, but element still there!");
#endif
}

/************************** LocMgr: MESSAGING *************************/
/// Deliver message to this element, going via the scheduler if local
/// @return 0 if object local, 1 if not
int CkLocMgr::deliverMsg(CkArrayMessage *msg, CkArrayID mgr, CmiUInt8 id, const CkArrayIndex* idx, CkDeliver_t type, int opts) {
  CkLocRec *rec = elementNrec(id);

#if CMK_LBDB_ON && 0
  LDObjid ldid = idx2LDObjid(idx);
#if CMK_GLOBAL_LOCATION_UPDATE
  ldid.locMgrGid = thisgroup.idx;
#endif        
  if (type==CkDeliver_queue && !(opts & CK_MSG_LB_NOTRACE) && the_lbdb->CollectingCommStats())
    the_lbdb->Send(myLBHandle
                   , ldid
                   , UsrToEnv(msg)->getTotalsize()
                   , lastKnown(id)
                   , 1);
#endif

  // Known, remote location or unknown location
  if (rec == NULL)
  {
    if (opts & CK_MSG_KEEP)
      msg = (CkArrayMessage *)CkCopyMsg((void **)&msg);
    // known location
    int destPE = whichPE(id);
    if (destPE != -1)
    {
      if((!CmiNodeAlive(destPE) && destPE != allowMessagesOnly)){
        CkAbort("Cannot send to a chare on a dead node");
      }
      msg->array_hops()++;
      CkArrayManagerDeliver(destPE,msg,opts);
      return true;
    }
    // unknown location
    deliverUnknown(msg,idx,type,opts);
    return true;
  }

  // Send via the msg q
  if (type==CkDeliver_queue)
  {
    if (opts & CK_MSG_KEEP)
      msg = (CkArrayMessage *)CkCopyMsg((void **)&msg);
    CkArrayManagerDeliver(CkMyPe(),msg,opts);
    return true;
  }

  ///@todo: Isnt there a chance that a local record exists, but all the managers are not created yet?
  /// in which case it should be buffered in bufferedMsgs
  CkMigratable *obj = managers[UsrToEnv(msg)->getArrayMgr()]->lookup(id);
  if (obj==NULL) {//That sibling of this object isn't created yet!
    if (opts & CK_MSG_KEEP)
      msg = (CkArrayMessage *)CkCopyMsg((void **)&msg);
    if (msg->array_ifNotThere()!=CkArray_IfNotThere_buffer)
      return demandCreateElement(msg, rec->getIndex(), CkMyPe(),type);
    else { // BUFFERING message for nonexistent element
      bufferedShadowElemMsgs[id].push_back(msg);
      return true;
    }
  }
        
  if (msg->array_hops()>1)
    multiHop(msg);
#if CMK_LBDB_ON
  // if there is a running obj being measured, stop it temporarily
  LDObjHandle objHandle;
  bool wasAnObjRunning = false;
  if (wasAnObjRunning = the_lbdb->RunningObject(&objHandle))
    the_lbdb->ObjectStop(objHandle);
#endif
  // Finally, call the entry method
  bool result = ((CkLocRec*)rec)->invokeEntry(obj,(void *)msg,msg->array_ep(),!(opts & CK_MSG_KEEP));
#if CMK_LBDB_ON
  if (wasAnObjRunning) the_lbdb->ObjectStart(objHandle);
#endif
  return result;
}

void CkLocMgr::sendMsg(CkArrayMessage *msg, CkArrayID mgr, const CkArrayIndex &idx, CkDeliver_t type, int opts) {
  CK_MAGICNUMBER_CHECK
  DEBS((AA "send %s\n" AB,idx2str(idx)));
  envelope *env = UsrToEnv(msg);
  env->setMsgtype(ForArrayEltMsg);

  if (type==CkDeliver_queue)
    _TRACE_CREATION_DETAILED(env, msg->array_ep());

  CmiUInt8 id;
  if (lookupID(idx, id)) {
    env->setRecipientID(ck::ObjID(mgr, id));
    deliverMsg(msg, mgr, id, &idx, type, opts);
    return;
  }

  env->setRecipientID(ck::ObjID(mgr, 0));

  int home = homePe(idx);
  if (home != CkMyPe()) {
    if (bufferedIndexMsgs.find(idx) == bufferedIndexMsgs.end())
      thisProxy[home].requestLocation(idx, CkMyPe(), false, msg->array_ifNotThere(), _entryTable[env->getEpIdx()]->chareIdx, mgr);
    bufferedIndexMsgs[idx].push_back(msg);

    return;
  }

  // We are the home, and there's no ID for this index yet - i.e. its
  // construction hasn't reached us yet.
  if (managers.find(mgr) == managers.end()) {
    // Even the manager for this array hasn't been constructed here yet
    if (CkInRestarting()) {
      // during restarting, this message should be ignored
      delete msg;
    } else {
      // Eventually, the manager will be created, and the element inserted, and
      // it will get pulled back out
      // 
      // XXX: Is demand creation ever possible in this case? I don't see why not
      bufferedIndexMsgs[idx].push_back(msg);
    }
    return;
  }

  // Copy the msg, if nokeep
  if (opts & CK_MSG_KEEP)
    msg = (CkArrayMessage *)CkCopyMsg((void **)&msg);

  // Buffer the msg
  bufferedIndexMsgs[idx].push_back(msg);

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
  envelope *env = UsrToEnv(msg);
  env->sender = CpvAccess(_currentObj)->mlogData->objID;
#endif

  // If requested, demand-create the element:
  if (msg->array_ifNotThere()!=CkArray_IfNotThere_buffer) {
    demandCreateElement(msg, idx, -1, type);
  }
}

/// This index is not hashed-- somehow figure out what to do.
void CkLocMgr::deliverUnknown(CkArrayMessage *msg, const CkArrayIndex* idx, CkDeliver_t type, int opts)
{
  CK_MAGICNUMBER_CHECK
  CmiUInt8 id = msg->array_element_id();
  int home;
  if (idx) home = homePe(*idx);
  else home = homePe(id);

  if (home != CkMyPe()) {// Forward the message to its home processor
    id2pe[id] = home;
    if (UsrToEnv(msg)->getTotalsize() < _messageBufferingThreshold) {
      DEBM((AA "Forwarding message for unknown %u to home %d \n" AB, id, home));
      msg->array_hops()++;
      CkArrayManagerDeliver(home, msg, opts);
    } else {
      DEBM((AA "Buffering message for unknown %u, home %d \n" AB, id, home));
      if (bufferedRemoteMsgs.find(id) == bufferedRemoteMsgs.end())
        thisProxy[home].requestLocation(id, CkMyPe(), false);
      bufferedRemoteMsgs[id].push_back(msg);
    }
  } else { // We *are* the home processor:
    //Check if the element's array manager has been registered yet:
    //No manager yet-- postpone the message (stupidly)
    if (managers.find(UsrToEnv((void*)msg)->getArrayMgr()) == managers.end()) {
      if (CkInRestarting()) {
        // during restarting, this message should be ignored
        delete msg;
      } else {
        CkArrayManagerDeliver(CkMyPe(),msg);
      }
    } else { // Has a manager-- must buffer the message
      // Copy the msg, if nokeep
      if (opts & CK_MSG_KEEP)
        msg = (CkArrayMessage *)CkCopyMsg((void **)&msg);
      // Buffer the msg
      bufferedMsgs[id].push_back(msg);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
      envelope *env = UsrToEnv(msg);
      env->sender = CpvAccess(_currentObj)->mlogData->objID;
#endif
      // If requested, demand-create the element:
      if (msg->array_ifNotThere()!=CkArray_IfNotThere_buffer) {
        CkAbort("Demand creation of elements is currently unimplemented");
      }
    }
  }
}

void CkLocMgr::demandCreateElement(const CkArrayIndex &idx, int chareType, int onPe, CkArrayID mgr)
{
  int ctor=_chareTable[chareType]->getDefaultCtor();
  if (ctor==-1) CkAbort("Can't create array element to handle message--\n"
                        "The element has no default constructor in the .ci file!\n");

  //Find the manager and build the element
  DEBC((AA "Demand-creating element %s on pe %d\n" AB,idx2str(idx),onPe));
  inform(idx, getNewObjectID(idx), onPe);
  managers[mgr]->demandCreateElement(idx, onPe, ctor, CkDeliver_inline);
}

bool CkLocMgr::demandCreateElement(CkArrayMessage *msg, const CkArrayIndex &idx, int onPe,CkDeliver_t type)
{
	CK_MAGICNUMBER_CHECK
	int chareType=_entryTable[msg->array_ep()]->chareIdx;
	int ctor=_chareTable[chareType]->getDefaultCtor();
	if (ctor==-1) CkAbort("Can't create array element to handle message--\n"
			      "The element has no default constructor in the .ci file!\n");
	if (onPe==-1) 
	{ //Decide where element needs to live
		if (msg->array_ifNotThere()==CkArray_IfNotThere_createhere) 
			onPe=UsrToEnv(msg)->getsetArraySrcPe();
		else //Createhome
			onPe=homePe(idx);
	}
	
	//Find the manager and build the element
	DEBC((AA "Demand-creating element %s on pe %d\n" AB,idx2str(idx),onPe));
	return managers[UsrToEnv((void *)msg)->getArrayMgr()]
        ->demandCreateElement(idx,onPe,ctor,type);
}

//This message took several hops to reach us-- fix it
void CkLocMgr::multiHop(CkArrayMessage *msg)
{
	CK_MAGICNUMBER_CHECK
	int srcPe=msg->array_getSrcPe();
	if (srcPe==CkMyPe())
          DEB((AA "Odd routing: local element %u is %d hops away!\n" AB, msg->array_element_id(),msg->array_hops()));
	else
	{//Send a routing message letting original sender know new element location
          DEBS((AA "Sending update back to %d for element %u\n" AB, srcPe, msg->array_element_id()));
          thisProxy[srcPe].updateLocation(msg->array_element_id(), CkMyPe());
	}
}

void CkLocMgr::checkInBounds(const CkArrayIndex &idx)
{
#if CMK_ERROR_CHECKING
  if (bounds.nInts > 0) {
    CkAssert(idx.dimension == bounds.dimension);
    bool shorts = idx.dimension > 3;

    for (int i = 0; i < idx.dimension; ++i) {
      unsigned int thisDim = shorts ? idx.indexShorts[i] : idx.index[i];
      unsigned int thatDim = shorts ? bounds.indexShorts[i] : bounds.index[i];
      CkAssert(thisDim < thatDim);
    }
  }
#endif
}

/************************** LocMgr: ITERATOR *************************/
CkLocation::CkLocation(CkLocMgr *mgr_, CkLocRec *rec_)
	:mgr(mgr_), rec(rec_) {}
	
const CkArrayIndex &CkLocation::getIndex(void) const {
	return rec->getIndex();
}

CmiUInt8 CkLocation::getID() const {
	return rec->getID();
}

void CkLocation::destroyAll() {
	mgr->callMethod(rec, &CkMigratable::ckDestroy);
}

void CkLocation::pup(PUP::er &p) {
	mgr->pupElementsFor(p,rec,CkElementCreation_migrate);
}

CkLocIterator::~CkLocIterator() {}

/// Iterate over our local elements:
void CkLocMgr::iterate(CkLocIterator &dest) {
  //Poke through the hash table for local ArrayRecs.
  void *objp;
  CkHashtableIterator *it=hash.iterator();
  CmiImmediateLock(hashImmLock);

  while (NULL!=(objp=it->next())) {
    CkLocRec *rec=*(CkLocRec **)objp;
    CkLocation loc(this,(CkLocRec *)rec);
    dest.addLocation(loc);
  }
  CmiImmediateUnlock(hashImmLock);
  delete it;
}




/************************** LocMgr: MIGRATION *************************/
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void CkLocMgr::pupElementsFor(PUP::er &p,CkLocRec *rec,
        CkElementCreation_t type, bool create, int dummy)
{
    p.comment("-------- Array Location --------");
    CkVec<CkMigratable *> dummyElts;

    for (std::map<CkArrayID, CkArray*>::iterator itr = managers.begin(); itr != managers.end(); ++itr) {
        int elCType;
        if (!p.isUnpacking())
        { //Need to find the element's existing type
            CkMigratable *elt = itr->second->getEltFromArrMgr(rec->getIndex());
            if (elt) elCType=elt->ckGetChareType();
            else elCType=-1; //Element hasn't been created
        }
        p(elCType);
        if (p.isUnpacking() && elCType!=-1) {
            CkMigratable *elt = itr->second->allocateMigrated(elCType,rec->getIndex(),type);
            int migCtorIdx=_chareTable[elCType]->getMigCtor();
                if(!dummy){
			if(create)
                    		if (!addElementToRec(rec, itr->second, elt, migCtorIdx, NULL)) return;
 				}else{
                    CkMigratable_initInfo &i=CkpvAccess(mig_initInfo);
                    i.locRec=rec;
                    i.chareType=_entryTable[migCtorIdx]->chareIdx;
                    dummyElts.push_back(elt);
                    if (!rec->invokeEntry(elt,NULL,migCtorIdx,true)) return ;
                }
        }
    }
    if(!dummy){
        for (std::map<CkArrayID, CkArray*>::iterator itr = managers.begin(); itr != managers.end(); ++itr) {
            CkMigratable *elt = itr->second->getEltFromArrMgr(rec->getIndex());
            if (elt!=NULL)
                {
                       elt->virtual_pup(p);
                }
        }
    }else{
            for(int i=0;i<dummyElts.size();i++){
                CkMigratable *elt = dummyElts[i];
                if (elt!=NULL){
            elt->virtual_pup(p);
        		}
                delete elt;
            }
            for (std::map<CkArrayID, CkArray*>::iterator itr = managers.begin(); itr != managers.end(); ++itr) {
                itr->second->eraseEltFromArrMgr(rec->getIndex());
            }
    }
}
#else
void CkLocMgr::pupElementsFor(PUP::er &p,CkLocRec *rec,
		CkElementCreation_t type,bool rebuild)
{
	p.comment("-------- Array Location --------");

	//First pup the element types
	// (A separate loop so ckLocal works even in element pup routines)
    for (std::map<CkArrayID, CkArray*>::iterator itr = managers.begin(); itr != managers.end(); ++itr) {
		int elCType;
		if (!p.isUnpacking())
		{ //Need to find the element's existing type
			CkMigratable *elt = itr->second->getEltFromArrMgr(rec->getID());
			if (elt) elCType=elt->ckGetChareType();
			else elCType=-1; //Element hasn't been created
		}
		p(elCType);
		if (p.isUnpacking() && elCType!=-1) {
			//Create the element
			CkMigratable *elt = itr->second->allocateMigrated(elCType, type);
			int migCtorIdx=_chareTable[elCType]->getMigCtor();
			//Insert into our tables and call migration constructor
			if (!addElementToRec(rec,itr->second,elt,migCtorIdx,NULL)) return;
		}
	}
	//Next pup the element data
    for (std::map<CkArrayID, CkArray*>::iterator itr = managers.begin(); itr != managers.end(); ++itr) {
		CkMigratable *elt = itr->second->getEltFromArrMgr(rec->getID());
		if (elt!=NULL)
                {
                        elt->virtual_pup(p);
#if CMK_ERROR_CHECKING
                        if (p.isUnpacking()) elt->sanitycheck();
#endif
                }
	}
#if CMK_MEM_CHECKPOINT
	if(rebuild){
	  ArrayElement *elt;
	  CkVec<CkMigratable *> list;
	  migratableList(rec, list);
	  CmiAssert(list.length() > 0);
	  for (int l=0; l<list.length(); l++) {
		//    reset, may not needed now
		// for now.
		for (int i=0; i<CK_ARRAYLISTENER_MAXLEN; i++) {
			ArrayElement * elt = (ArrayElement *)list[l];
		  contributorInfo *c=(contributorInfo *)&elt->listenerData[i];
		  if (c) c->redNo = 0;
		}
	  }
		
	}
#endif
}
#endif

/// Call this member function on each element of this location:
void CkLocMgr::callMethod(CkLocRec *rec,CkMigratable_voidfn_t fn)
{
    for (std::map<CkArrayID, CkArray*>::iterator itr = managers.begin(); itr != managers.end(); ++itr) {
		CkMigratable *el = itr->second->getEltFromArrMgr(rec->getID());
		if (el) (el->* fn)();
	}
}

/// Call this member function on each element of this location:
void CkLocMgr::callMethod(CkLocRec *rec,CkMigratable_voidfn_arg_t fn,     void * data)
{
    for (std::map<CkArrayID, CkArray*>::iterator itr = managers.begin(); itr != managers.end(); ++itr) {
		CkMigratable *el = itr->second->getEltFromArrMgr(rec->getID());
		if (el) (el->* fn)(data);
	}
}

/// return a list of migratables in this local record
void CkLocMgr::migratableList(CkLocRec *rec, CkVec<CkMigratable *> &list)
{
        for (std::map<CkArrayID, CkArray*>::iterator itr = managers.begin(); itr != managers.end(); ++itr) {
                CkMigratable *elt = itr->second->getEltFromArrMgr(rec->getID());
                if (elt) list.push_back(elt);
        }
}

/// Migrate this local element away to another processor.
void CkLocMgr::emigrate(CkLocRec *rec,int toPe)
{
	CK_MAGICNUMBER_CHECK
	if (toPe==CkMyPe()) return; //You're already there!
	/*
		FAULT_EVAC
		if the toProcessor is already marked as invalid, dont emigrate
		Shouldn't happen but might
	*/
	if(!CmiNodeAlive(toPe)){
		return;
	}
	CkArrayIndex idx=rec->getIndex();
        CmiUInt8 id = rec->getID();

#if CMK_OUT_OF_CORE
	/* Load in any elements that are out-of-core */
    for (std::map<CkArrayID, CkArray*>::iterator itr = managers.begin(); itr != managers.end(); ++itr) {
		CkMigratable *el = itr->second->getEltFromArrMgr(rec->getIndex());
		if (el) if (!el->isInCore) CooBringIn(el->prefetchObjID);
	}
#endif

	//Let all the elements know we're leaving
	callMethod(rec,&CkMigratable::ckAboutToMigrate);
	/*EVAC*/

//First pass: find size of migration message
	size_t bufSize;
	{
		PUP::sizer p;
		pupElementsFor(p,rec,CkElementCreation_migrate);
		bufSize=p.size(); 
	}

//Allocate and pack into message
	CkArrayElementMigrateMessage *msg = new (bufSize, 0) CkArrayElementMigrateMessage(idx, id,
#if CMK_LBDB_ON
		rec->isAsyncMigrate(),
#else
		false,
#endif
		bufSize, managers.size(), rec->isBounced());

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_)) 
    msg->gid = ckGetGroupID();
#endif
	{
		PUP::toMem p(msg->packData); 
		p.becomeDeleting(); 
		pupElementsFor(p,rec,CkElementCreation_migrate);
		if (p.size()!=bufSize) {
			CkError("ERROR! Array element claimed it was %d bytes to a "
				"sizing PUP::er, but copied %d bytes into the packing PUP::er!\n",
				bufSize,p.size());
			CkAbort("Array element's pup routine has a direction mismatch.\n");
		}
	}

	DEBM((AA "Migrated index size %s to %d \n" AB,idx2str(idx),toPe));	

//#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
//#if defined(_FAULT_MLOG_)
//    sendMlogLocation(toPe,UsrToEnv(msg));
//#else
	//Send off message and delete old copy
	thisProxy[toPe].immigrate(msg);
//#endif

	duringMigration=true;
	delete rec; //Removes elements, hashtable entries, local index
	
	
	duringMigration=false;
	//The element now lives on another processor-- tell ourselves and its home
	inform(idx, id, toPe);
//#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))    
//#if !defined(_FAULT_MLOG_)    
	informHome(idx,toPe);
//#endif

#if !CMK_LBDB_ON && CMK_GLOBAL_LOCATION_UPDATE
        DEBM((AA "Global location update. idx %s " 
              "assigned to %d \n" AB,idx2str(idx),toPe));
        thisProxy.updateLocation(idx, toPe);                        
#endif

	CK_MAGICNUMBER_CHECK
}

#if CMK_LBDB_ON
void CkLocMgr::informLBPeriod(CkLocRec *rec, int lb_ideal_period) {
	callMethod(rec,&CkMigratable::recvLBPeriod, (void *)&lb_ideal_period);
}

void CkLocMgr::metaLBCallLB(CkLocRec *rec) {
	callMethod(rec, &CkMigratable::metaLBCallLB);
}
#endif

/**
  Migrating array element is arriving on this processor.
*/
void CkLocMgr::immigrate(CkArrayElementMigrateMessage *msg)
{
	const CkArrayIndex &idx=msg->idx;
		
	PUP::fromMem p(msg->packData); 
	
	if (msg->nManagers < managers.size())
		CkAbort("Array element arrived from location with fewer managers!\n");
	if (msg->nManagers > managers.size()) {
		//Some array managers haven't registered yet-- throw it back
		DEBM((AA "Busy-waiting for array registration on migrating %s\n" AB,idx2str(idx)));
		thisProxy[CkMyPe()].immigrate(msg);
		return;
	}

        idx2id[idx] = msg->id;

	//Create a record for this element
//#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))    
//#if !defined(_FAULT_MLOG_)     
	CkLocRec *rec=createLocal(idx,true,msg->ignoreArrival,false /* home told on departure */ );
//#else
//    CkLocRec *rec=createLocal(idx,true,true,false /* home told on departure */ );
//#endif
	
	//Create the new elements as we unpack the message
	pupElementsFor(p,rec,CkElementCreation_migrate);
	if (p.size()!=msg->length) {
		CkError("ERROR! Array element claimed it was %d bytes to a"
			"packing PUP::er, but %d bytes in the unpacking PUP::er!\n",
			msg->length,p.size());
		CkError("(I have %d managers; he claims %d managers)\n",
			managers.size(), msg->nManagers);
		
		CkAbort("Array element's pup routine has a direction mismatch.\n");
	}
	/*
		FAULT_EVAC
			if this element came in as a result of being bounced off some other process,
			then it needs to be resumed. It is assumed that it was bounced because load 
			balancing caused it to move into a processor which later crashed
	*/
	if(msg->bounced){
		callMethod(rec,&CkMigratable::ResumeFromSync);
	}
	
	//Let all the elements know we've arrived
	callMethod(rec,&CkMigratable::ckJustMigrated);
	/*FAULT_EVAC
		If this processor has started evacuating array elements on it 
		dont let new immigrants in. If they arrive send them to what
		would be their correct homePE.
		Leave a record here mentioning the processor where it got sent
	*/
	
	if(CkpvAccess(startedEvac)){
		int newhomePE = getNextPE(idx);
		DEBM((AA "Migrated into failed processor index size %s resent to %d \n" AB,idx2str(idx),newhomePE));	
		int targetPE=getNextPE(idx);
		//set this flag so that load balancer is not informed when
		//this element migrates
		rec->AsyncMigrate(true);
		rec->Bounced(true);
		emigrate(rec,targetPE);
	}

	delete msg;
}

void CkLocMgr::restore(const CkArrayIndex &idx, CmiUInt8 id, PUP::er &p)
{
	idx2id[idx] = id;

	//This is in broughtIntoMem during out-of-core emulation in BigSim,
	//informHome should not be called since such information is already
	//immediately updated real migration
#if CMK_ERROR_CHECKING
	if(_BgOutOfCoreFlag!=2)
	    CmiAbort("CkLocMgr::restore should only be used in out-of-core emulation for BigSim and be called when object is brought into memory!\n");
#endif
	CkLocRec *rec=createLocal(idx,false,false,false);
	
	//BIGSIM_OOC DEBUGGING
	//CkPrintf("Proc[%d]: Registering element %s with LDB\n", CkMyPe(), idx2str(idx));

	//Create the new elements as we unpack the message
	pupElementsFor(p,rec,CkElementCreation_restore);

	callMethod(rec,&CkMigratable::ckJustRestored);
}


/// Insert and unpack this array element from this checkpoint (e.g., from CkLocation::pup)
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void CkLocMgr::resume(const CkArrayIndex &idx, CmiUInt8 id, PUP::er &p, bool create, int dummy)
{
	idx2id[idx] = id;

	CkLocRec *rec;

	if(create){
		rec = createLocal(idx,false,false,true && !dummy /* home doesn't know yet */,dummy );
	}else{
		rec = elementNrec(idx);
		if(rec == NULL) 
			CmiAbort("Local object not found");
	}
        
    pupElementsFor(p,rec,CkElementCreation_resume,create,dummy);

    if(!dummy){
        callMethod(rec,&CkMigratable::ckJustMigrated);
    }
}
#else
void CkLocMgr::resume(const CkArrayIndex &idx, CmiUInt8 id, PUP::er &p, bool notify,bool rebuild)
{
	idx2id[idx] = id;

	CkLocRec *rec=createLocal(idx,false,false,notify /* home doesn't know yet */ );

	//Create the new elements as we unpack the message
	pupElementsFor(p,rec,CkElementCreation_resume,rebuild);

	callMethod(rec,&CkMigratable::ckJustMigrated);
}
#endif

/********************* LocMgr: UTILITY ****************/
void CkMagicNumber_impl::badMagicNumber(
	int expected,const char *file,int line,void *obj) const
{
	CkError("FAILURE on pe %d, %s:%d> Expected %p's magic number "
		"to be 0x%08x; but found 0x%08x!\n", CkMyPe(),file,line,obj,
		expected, magic);
	CkAbort("Bad magic number detected!  This implies either\n"
		"the heap or a message was corrupted!\n");
}
CkMagicNumber_impl::CkMagicNumber_impl(int m) :magic(m) { }

int CkLocMgr::whichPE(const CkArrayIndex &idx) const
{
  CmiUInt8 id;
  if (!lookupID(idx, id))
    return -1;

  std::map<CmiUInt8, int>::const_iterator itr = id2pe.find(id);
  return (itr != id2pe.end() ? itr->second : -1);
}

int CkLocMgr::whichPE(const CmiUInt8 id) const
{
  std::map<CmiUInt8, int>::const_iterator itr = id2pe.find(id);
  return (itr != id2pe.end() ? itr->second : -1);
}

//"Last-known" location (returns a processor number)
int CkLocMgr::lastKnown(const CkArrayIndex &idx) {
	CkLocMgr *vthis=(CkLocMgr *)this;//Cast away "const"
	int pe = whichPE(idx);
	if (pe==-1) return homePe(idx);
	else{
		if(!CmiNodeAlive(pe)){
			CkAbort("Last known PE is no longer alive");
		}
		return pe;
	}	
}

//"Last-known" location (returns a processor number)
int CkLocMgr::lastKnown(CmiUInt8 id) {
  int pe = whichPE(id);
  if (pe==-1) return homePe(id);
  else{
    if(!CmiNodeAlive(pe)){
      CkAbort("Last known PE is no longer alive");
    }
    return pe;
  }	
}

/// Return true if this array element lives on another processor
bool CkLocMgr::isRemote(const CkArrayIndex &idx,int *onPe) const
{
    int pe = whichPE(idx);
    /* not definitely a remote element */
    if (pe == -1 || pe == CkMyPe())
        return false;
    // element is indeed remote
    *onPe = pe;
    return true;
}

static const char *rec2str[]={
    "base (INVALID)",//Base class (invalid type)
    "local",//Array element that lives on this Pe
};

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void CkLocMgr::setDuringMigration(bool _duringMigration){
    duringMigration = _duringMigration;
}
#endif

void CkLocMgr::setDuringDestruction(bool _duringDestruction) {
  duringDestruction = _duringDestruction;
}

//Add given element array record at idx, replacing the existing record
void CkLocMgr::insertRec(CkLocRec *rec, const CmiUInt8 &id) {
    CkLocRec *old_rec = elementNrec(id);

    CmiImmediateLock(hashImmLock);
    hash.put(id) = rec;
    CmiImmediateUnlock(hashImmLock);

    delete old_rec;
}

//Call this on an unrecognized array index
static void abort_out_of_bounds(const CkArrayIndex &idx)
{
  CkPrintf("ERROR! Unknown array index: %s\n",idx2str(idx));
  CkAbort("Array index out of bounds\n");
}

//Look up array element in hash table.  Index out-of-bounds if not found.
CkLocRec *CkLocMgr::elementRec(const CkArrayIndex &idx) {
#if ! CMK_ERROR_CHECKING
//Assume the element will be found
  return hash.getRef(lookupID(idx));
#else
//Include an out-of-bounds check if the element isn't found
  CmiUInt8 id;
  CkLocRec *rec;
  if (lookupID(idx, id) && (rec = elementNrec(id))) {
	return rec;
  } else {
	if (rec==NULL) abort_out_of_bounds(idx);
  }
#endif
}

//Look up array element in hash table.  Return NULL if not there.
CkLocRec *CkLocMgr::elementNrec(const CmiUInt8 id) {
  return hash.get(id);
}

struct LocalElementCounter :  public CkLocIterator
{
    unsigned int count;
    LocalElementCounter() : count(0) {}
    void addLocation(CkLocation &loc)
	{ ++count; }
};

unsigned int CkLocMgr::numLocalElements()
{
    LocalElementCounter c;
    iterate(c);
    return c.count;
}


/********************* LocMgr: LOAD BALANCE ****************/

#if !CMK_LBDB_ON
//Empty versions of all load balancer calls
void CkLocMgr::initLB(CkGroupID lbdbID_, CkGroupID metalbID_) {}
void CkLocMgr::startInserting(void) {}
void CkLocMgr::doneInserting(void) {}
void CkLocMgr::dummyAtSync(void) {}
#endif


#if CMK_LBDB_ON
void CkLocMgr::initLB(CkGroupID lbdbID_, CkGroupID metalbID_)
{ //Find and register with the load balancer
	the_lbdb = (LBDatabase *)CkLocalBranch(lbdbID_);
	if (the_lbdb == 0)
		CkAbort("LBDatabase not yet created?\n");
	DEBL((AA "Connected to load balancer %p\n" AB,the_lbdb));
	the_metalb = (MetaBalancer *)CkLocalBranch(metalbID_);
	if (the_metalb == 0)
		CkAbort("MetaBalancer not yet created?\n");

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
	myLBHandle = the_lbdb->RegisterOM(myId,this,myCallbacks);

	// Tell the lbdb that I'm registering objects
	the_lbdb->RegisteringObjects(myLBHandle);

	/*Set up the dummy barrier-- the load balancer needs
	  us to call Registering/DoneRegistering during each AtSync,
	  and this is the only way to do so.
	*/
	lbBarrierReceiver = the_lbdb->AddLocalBarrierReceiver(
		(LDBarrierFn)staticRecvAtSync,(void*)(this));
	dummyBarrierHandle = the_lbdb->AddLocalBarrierClient(
		(LDResumeFn)staticDummyResumeFromSync,(void*)(this));
	dummyAtSync();
}
void CkLocMgr::dummyAtSync(void)
{
	DEBL((AA "dummyAtSync called\n" AB));
	the_lbdb->AtLocalBarrier(dummyBarrierHandle);
}

void CkLocMgr::staticDummyResumeFromSync(void* data)
{      ((CkLocMgr*)data)->dummyResumeFromSync(); }
void CkLocMgr::dummyResumeFromSync()
{
	DEBL((AA "DummyResumeFromSync called\n" AB));
	the_lbdb->DoneRegisteringObjects(myLBHandle);
	dummyAtSync();
}
void CkLocMgr::staticRecvAtSync(void* data)
{      ((CkLocMgr*)data)->recvAtSync(); }
void CkLocMgr::recvAtSync()
{
	DEBL((AA "recvAtSync called\n" AB));
	the_lbdb->RegisteringObjects(myLBHandle);
}

void CkLocMgr::startInserting(void)
{
	the_lbdb->RegisteringObjects(myLBHandle);
}
void CkLocMgr::doneInserting(void)
{
	the_lbdb->DoneRegisteringObjects(myLBHandle);
}
#endif

#include "CkLocation.def.h"



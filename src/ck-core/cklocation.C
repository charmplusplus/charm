/** \file cklocation.C
 *  \addtogroup CkArrayImpl
 *
 *  The location manager keeps track of an indexed set of migratable objects.
 *  It is used by the array manager to locate array elements, interact with the
 *  load balancer, and perform migrations.
 *
 *  Orion Sky Lawlor, olawlor@acm.org 9/29/2001
 */

#include "charm++.h"
#include "register.h"
#include "ck.h"
#include "trace.h"
#include "TopoManager.h"

#include<sstream>

#if CMK_LBDB_ON
#include "LBDatabase.h"
#endif // CMK_LBDB_ON

#if CMK_GRID_QUEUE_AVAILABLE
CpvExtern(void *, CkGridObject);
#endif

static const char *idx2str(const CkArrayMessage *m) {
  return idx2str(((CkArrayMessage *)m)->array_index());
}

#define ARRAY_DEBUG_OUTPUT 0

#if ARRAY_DEBUG_OUTPUT 
#   define DEB(x) CkPrintf x  //General debug messages
#   define DEBI(x) CkPrintf x  //Index debug messages
#   define DEBC(x) CkPrintf x  //Construction debug messages
#   define DEBS(x) CkPrintf x  //Send/recv/broadcast debug messages
#   define DEBM(x) CkPrintf x  //Migration debug messages
#   define DEBL(x) CkPrintf x  //Load balancing debug messages
#   define DEBK(x) //CkPrintf x  //Spring Cleaning debug messages
#   define DEBB(x) CkPrintf x  //Broadcast debug messages
#   define AA "LocMgr on %d: "
#   define AB ,CkMyPe()
#   define DEBUG(x) CkPrintf x
#else
#   define DEB(X) /*CkPrintf x*/
#   define DEBI(X) /*CkPrintf x*/
#   define DEBC(X) /*CkPrintf x*/
#   define DEBS(x) /*CkPrintf x*/
#   define DEBM(X) /*CkPrintf x*/
#   define DEBL(X) /*CkPrintf x*/
#   define DEBK(x) /*CkPrintf x*/
#   define DEBB(x) /*CkPrintf x*/
#   define str(x) /**/
#   define DEBUG(x)   /**/
#endif

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
  return r;
}
#endif

/*********************** Array Messages ************************/
CkArrayIndex &CkArrayMessage::array_index(void)
{
    return UsrToEnv((void *)this)->getsetArrayIndex();
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
int CkArrayMap::registerArray(CkArrayIndex& numElements,CkArrayID aid)
{return 0;}

#define CKARRAYMAP_POPULATE_INITIAL(POPULATE_CONDITION) \
        int i; \
	for (int i1=0; i1<numElements.data()[0]; i1++) { \
          if (numElements.dimension == 1) { \
            /* Make 1D indices */ \
            i = i1; \
            CkArrayIndex1D idx(i1); \
            if (POPULATE_CONDITION) \
              mgr->insertInitial(idx,CkCopyMsg(&ctorMsg)); \
          } else { \
            /* higher dimensionality */ \
            for (int i2=0; i2<numElements.data()[1]; i2++) { \
              if (numElements.dimension == 2) { \
                /* Make 2D indices */ \
                i = i1 * numElements.data()[1] + i2; \
                CkArrayIndex2D idx(i1, i2); \
                if (POPULATE_CONDITION) \
                  mgr->insertInitial(idx,CkCopyMsg(&ctorMsg)); \
              } else { \
                /* higher dimensionality */ \
                CkAssert(numElements.dimension == 3); \
                for (int i3=0; i3<numElements.data()[2]; i3++) { \
                  /* Make 3D indices */ \
                  i = (i1 * numElements.data()[1] + i2) * numElements.data()[2] + i3; \
                  CkArrayIndex3D idx(i1, i2, i3 ); \
                  if (POPULATE_CONDITION) \
                    mgr->insertInitial(idx,CkCopyMsg(&ctorMsg)); \
                } \
              } \
            } \
          } \
	}

void CkArrayMap::populateInitial(int arrayHdl,CkArrayIndex& numElements,void *ctorMsg,CkArrMgr *mgr)
{
	if (numElements.nInts==0) {
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
	  DEBC((AA"Creating RRMap\n"AB));
  }
  RRMap(CkMigrateMessage *m):CkArrayMap(m){}
  int procNum(int /*arrayHdl*/, const CkArrayIndex &i)
  {
#if 1
    if (i.nInts==1) {
      //Map 1D integer indices in simple round-robin fashion
      int ans= (i.data()[0])%CkNumPes();
      while(!CmiNodeAlive(ans) || (ans == CkMyPe() && CkpvAccess(startedEvac))){
        ans = (ans +1 )%CkNumPes();
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
		ans = (ans +1 )%CkNumPes();
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

  /** All processors are divided into two sets. Processors in the first set
   *  have one chare more than the processors in the second set. */

  arrayMapInfo(void) { }

  arrayMapInfo(CkArrayIndex& n) : _nelems(n), _numChares(0) {
    compute_binsize();
  }

  ~arrayMapInfo() {}
  
  void compute_binsize()
  {
    int numPes = CkNumPes();

    if (_nelems.nInts == 1) {
      _numChares = _nelems.data()[0];
    } else if (_nelems.nInts == 2) {
      _numChares = _nelems.data()[0] * _nelems.data()[1];
    } else if (_nelems.nInts == 3) {
      _numChares = _nelems.data()[0] * _nelems.data()[1] * _nelems.data()[2];
    }

    _remChares = _numChares % numPes;
    _binSizeFloor = (int)floor((double)_numChares/(double)numPes);
    _binSizeCeil = (int)ceil((double)_numChares/(double)numPes);
    _numFirstSet = _remChares * (_binSizeFloor + 1);
  }

  void pup(PUP::er& p){
    p|_nelems;
    p|_binSizeFloor;
    p|_binSizeCeil;
    p|_numChares;
    p|_remChares;
    p|_numFirstSet;
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
    DEBC((AA"Creating DefaultArrayMap\n"AB));
  }

  DefaultArrayMap(CkMigrateMessage *m) : RRMap(m){}

  int registerArray(CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx = amaps.size();
    amaps.resize(idx+1);
    amaps[idx] = new arrayMapInfo(numElements);
    return idx;
  }
 
  int procNum(int arrayHdl, const CkArrayIndex &i) {
    int flati;
    if (amaps[arrayHdl]->_nelems.nInts == 0) {
      return RRMap::procNum(arrayHdl, i);
    }

    if (i.nInts == 1) {
      flati = i.data()[0];
    } else if (i.nInts == 2) {
      flati = i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1];
    } else if (i.nInts == 3) {
      flati = (i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1]) * amaps[arrayHdl]->_nelems.data()[2] + i.data()[2];
    }
#if CMK_ERROR_CHECKING
    else {
      CkAbort("CkArrayIndex has more than 3 integers!");
    }
#endif

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
    DEBC((AA"Creating FastArrayMap\n"AB));
  }

  FastArrayMap(CkMigrateMessage *m) : DefaultArrayMap(m){}

  int registerArray(CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx;
    idx = DefaultArrayMap::registerArray(numElements, aid);

    return idx;
  }

  int procNum(int arrayHdl, const CkArrayIndex &i) {
    int flati;
    if (amaps[arrayHdl]->_nelems.nInts == 0) {
      return RRMap::procNum(arrayHdl, i);
    }

    if (i.nInts == 1) {
      flati = i.data()[0];
    } else if (i.nInts == 2) {
      flati = i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1];
    } else if (i.nInts == 3) {
      flati = (i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1]) * amaps[arrayHdl]->_nelems.data()[2] + i.data()[2];
    }
#if CMK_ERROR_CHECKING
    else {
      CkAbort("CkArrayIndex has more than 3 integers!");
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
    DEBC((AA"Creating ReadFileMap\n"AB));
  }

  ReadFileMap(CkMigrateMessage *m) : DefaultArrayMap(m){}

  int registerArray(CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx;
    idx = DefaultArrayMap::registerArray(numElements, aid);

    if(mapping.size() == 0) {
      int numChares;

      if (amaps[idx]->_nelems.nInts == 1) {
	numChares = amaps[idx]->_nelems.data()[0];
      } else if (amaps[idx]->_nelems.nInts == 2) {
	numChares = amaps[idx]->_nelems.data()[0] * amaps[idx]->_nelems.data()[1];
      } else if (amaps[idx]->_nelems.nInts == 3) {
	numChares = amaps[idx]->_nelems.data()[0] * amaps[idx]->_nelems.data()[1] * amaps[idx]->_nelems.data()[2];
      } else {
	CkAbort("CkArrayIndex has more than 3 integers!");
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

    if (i.nInts == 1) {
      flati = i.data()[0];
    } else if (i.nInts == 2) {
      flati = i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1];
    } else if (i.nInts == 3) {
      flati = (i.data()[0] * amaps[arrayHdl]->_nelems.data()[1] + i.data()[1]) * amaps[arrayHdl]->_nelems.data()[2] + i.data()[2];
    } else {
      CkAbort("CkArrayIndex has more than 3 integers!");
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
	DEBC((AA"Creating BlockMap\n"AB));
  }
  BlockMap(CkMigrateMessage *m):RRMap(m){ }
  void populateInitial(int arrayHdl,CkArrayIndex& numElements,void *ctorMsg,CkArrMgr *mgr){
	if (numElements.nInts==0) {
          CkFreeMsg(ctorMsg);
          return;
        }
	int thisPe=CkMyPe();
	int numPes=CkNumPes();
        int binSize;
        if (numElements.nInts == 1) {
          binSize = (int)ceil((double)numElements.data()[0]/(double)numPes);
        } else if (numElements.nInts == 2) {
          binSize = (int)ceil((double)(numElements.data()[0]*numElements.data()[1])/(double)numPes);
        } else if (numElements.nInts == 3) {
          binSize = (int)ceil((double)(numElements.data()[0]*numElements.data()[1]*numElements.data()[2])/(double)numPes);
        } else {
          CkAbort("CkArrayIndex has more than 3 integers!");
        }
        CKARRAYMAP_POPULATE_INITIAL(i/binSize==thisPe);

        /*
        CkArrayIndex idx;
	for (idx=numElements.begin(); idx<numElements; idx.getNext(numElements)) {
          //for (int i=0;i<numElements;i++) {
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
	  DEBC((AA"Creating CldMap\n"AB));
  }
  CldMap(CkMigrateMessage *m):CkArrayMap(m){}
  int homePe(int /*arrayHdl*/, const CkArrayIndex &i)
  {
    if (i.nInts==1) {
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
  void populateInitial(int arrayHdl,CkArrayIndex& numElements,void *ctorMsg,CkArrMgr *mgr)  {
        if (numElements.nInts==0) {
          CkFreeMsg(ctorMsg);
          return;
        }
        int thisPe=CkMyPe();
        int numPes=CkNumPes();
        //CkArrayIndex idx;

        CKARRAYMAP_POPULATE_INITIAL(i%numPes==thisPe);
	/*for (idx=numElements.begin(); idx<numElements; idx.getNext(numElements)) {
          //for (int i=0;i<numElements;i++)
                        if((idx.getRank(numElements))%numPes==thisPe)
                                mgr->insertInitial(CkArrayIndex1D(i),CkCopyMsg(&ctorMsg),0);
        }*/
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
	DEBC((AA"Creating ConfigurableRRMap\n"AB));
  }
  ConfigurableRRMap(CkMigrateMessage *m):RRMap(m){ }


  void populateInitial(int arrayHdl,CkArrayIndex& numElements,void *ctorMsg,CkArrMgr *mgr){
    // Try to load the configuration from command line argument
    CkAssert(haveConfigurableRRMap());
    ConfigurableRRMapLoader &loader =  CkpvAccess(myConfigRRMapState);
    if (numElements.nInts==0) {
      CkFreeMsg(ctorMsg);
      return;
    }
    int thisPe=CkMyPe();
    int maxIndex = numElements.data()[0];
    DEBUG(("[%d] ConfigurableRRMap: index=%d,%d,%d\n", CkMyPe(),(int)numElements.data()[0], (int)numElements.data()[1], (int)numElements.data()[2]));

    if (numElements.nInts != 1) {
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
   void distrib(int *speeds);
 public:
   arrInfo(void):_map(NULL){}
   arrInfo(CkArrayIndex& n, int *speeds)
   {
     _nelems = n;
     _map = new int[_nelems.getCombinedCount()];
     distrib(speeds);
   }
   ~arrInfo() { delete[] _map; }
   int getMap(const CkArrayIndex &i);
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
  if(i.nInts==1)
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
  int hdlr = CkRegisterHandler((CmiHandler)_speedHdlr);
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
    DEBC((AA"Creating PropMap\n"AB));
  }
  PropMap(CkMigrateMessage *m) {}
  int registerArray(CkArrayIndex& numElements,CkArrayID aid)
  {
    int idx = arrs.size();
    arrs.resize(idx+1);
    arrs[idx] = new arrInfo(numElements, speeds);
    return idx;
  }
  int procNum(int arrayHdl, const CkArrayIndex &i)
  {
    return arrs[arrayHdl]->getMap(i);
  }
  void pup(PUP::er& p){
    p|arrs;
  }
};

class CkMapsInit : public Chare
{
public:
	CkMapsInit(CkArgMsg *msg) {
		_defaultArrayMapID = CProxy_DefaultArrayMap::ckNew();
		_fastArrayMapID = CProxy_FastArrayMap::ckNew();
		delete msg;
	}

	CkMapsInit(CkMigrateMessage *m) {}
};

// given an envelope of a Charm msg, find the recipient object pointer
CkMigratable * CkArrayMessageObjectPtr(envelope *env) {
  if (env->getMsgtype()!=ForArrayEltMsg) return NULL;   // not an array msg

  CkArrayID aid = env->getsetArrayMgr();
  CkArray *mgr=(CkArray *)_localBranch(aid);
  if (mgr) {
    CkLocMgr *locMgr = mgr->getLocMgr();
    if (locMgr) {
      return locMgr->lookup(env->getsetArrayIndex(),aid);
    }
  }
  return NULL;
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
  elt->pup(p);

  //Call the element's destructor in-place (so pointer doesn't change)
  CkpvAccess(CkSaveRestorePrefetch)=1;
  elt->~CkMigratable(); //< because destuctor is virtual, destroys user class too.
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
  elt->myRec->invokeEntry(elt,(CkMigrateMessage *)0,ctorIdx,CmiTrue);
  CkpvAccess(CkSaveRestorePrefetch)=0;
  
  //Restore the element's data from disk:
  PUP::fromDisk p(swapfile);
  elt->pup(p);
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
	CkLocRec_local *locRec;
	int chareType;
	CmiBool forPrefetch; /* If true, this creation is only a prefetch restore-from-disk.*/
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
	isInCore=CmiTrue;
	if (CkpvAccess(CkSaveRestorePrefetch))
		return; /* Just restoring from disk--don't touch object */
	prefetchObjID=-1; //Unregistered
#endif
	myRec=i.locRec;
	thisIndexMax=myRec->getIndex();
	thisChareType=i.chareType;
	usesAtSync=CmiFalse;
	usesAutoMeasure=CmiTrue;
	barrierRegistered=CmiFalse;
  atsync_iteration = -1;
  //CkPrintf("%s in init and off\n", idx2str(thisIndexMax));
  local_state = OFF;
  prev_load = 0.0;
	/*
	FAULT_EVAC
	*/
	AsyncEvacuate(CmiTrue);
}

CkMigratable::CkMigratable(void) {
	DEBC((AA"In CkMigratable constructor\n"AB));
	commonInit();
}
CkMigratable::CkMigratable(CkMigrateMessage *m): Chare(m) {
	commonInit();
}

int CkMigratable::ckGetChareType(void) const {return thisChareType;}

void CkMigratable::pup(PUP::er &p) {
	DEBM((AA"In CkMigratable::pup %s\n"AB,idx2str(thisIndexMax)));
	Chare::pup(p);
	p|thisIndexMax;
	p(usesAtSync);
	p(usesAutoMeasure);
#if CMK_LBDB_ON 
	int readyMigrate;
	if (p.isPacking()) readyMigrate = myRec->isReadyMigrate();
	p|readyMigrate;
	if (p.isUnpacking()) myRec->ReadyMigrate(readyMigrate);
#endif
	if(p.isUnpacking()) barrierRegistered=CmiFalse;
	/*
		FAULT_EVAC
	*/
	p | asyncEvacuate;
	if(p.isUnpacking()){myRec->AsyncEvacuate(asyncEvacuate);}
	
	ckFinishConstruction();
}

void CkMigratable::ckDestroy(void) {
	DEBC((AA"In CkMigratable::ckDestroy %s\n"AB,idx2str(thisIndexMax)));
	myRec->destroy();
}

void CkMigratable::ckAboutToMigrate(void) { }
void CkMigratable::ckJustMigrated(void) { }
void CkMigratable::ckJustRestored(void) { }

CkMigratable::~CkMigratable() {
	DEBC((AA"In CkMigratable::~CkMigratable %s\n"AB,idx2str(thisIndexMax)));
#if CMK_OUT_OF_CORE
	isInCore=CmiFalse;
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
	  DEBL((AA"Removing barrier for element %s\n"AB,idx2str(thisIndexMax)));
	  if (usesAtSync)
		myRec->getLBDB()->RemoveLocalBarrierClient(ldBarrierHandle);
	  else
		myRec->getLBDB()->RemoveLocalBarrierReceiver(ldBarrierRecvHandle);
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

void CkMigratable::recvLBPeriod(void *data) {
  int lb_period = *((int *) data);
  //CkPrintf("--[pe %s] Received the LB Period %d current iter %d state %d\n",
   //   idx2str(thisIndexMax), lb_period, atsync_iteration, local_state);
  if (local_state == PAUSE) {
    if (atsync_iteration < lb_period) {
    //  CkPrintf("---[pe %s] pause and decided\n", idx2str(thisIndexMax));
      local_state = DECIDED;
      ResumeFromSync();
      return;
    }
   // CkPrintf("---[pe %s] load balance\n", idx2str(thisIndexMax));
    local_state = LOAD_BALANCE;

    local_state = OFF;
    atsync_iteration = -1;
    prev_load = 0.0;

    myRec->getLBDB()->AtLocalBarrier(ldBarrierHandle);
    return;
  }
 // CkPrintf("---[pe %s] decided\n", idx2str(thisIndexMax));
  local_state = DECIDED;
}

void CkMigratable::ckFinishConstruction(void)
{
//	if ((!usesAtSync) || barrierRegistered) return;
	myRec->setMeasure(usesAutoMeasure);
	if (barrierRegistered) return;
	DEBL((AA"Registering barrier client for %s\n"AB,idx2str(thisIndexMax)));
        if (usesAtSync)
	  ldBarrierHandle = myRec->getLBDB()->AddLocalBarrierClient(
		(LDBarrierFn)staticResumeFromSync,(void*)(this));
        else
	  ldBarrierRecvHandle = myRec->getLBDB()->AddLocalBarrierReceiver(
		(LDBarrierFn)staticResumeFromSync,(void*)(this));
	barrierRegistered=CmiTrue;
}

void CkMigratable::AtSync(int waitForMigration)
{
	if (!usesAtSync)
		CkAbort("You must set usesAtSync=CmiTrue in your array element constructor to use AtSync!\n");
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        mlogData->toResumeOrNot=1;
#endif
	myRec->AsyncMigrate(!waitForMigration);
	if (waitForMigration) ReadyMigrate(CmiTrue);
	ckFinishConstruction();
  DEBL((AA"Element %s going to sync\n"AB,idx2str(thisIndexMax)));
  // model-based load balancing, ask user to provide cpu load
  if (usesAutoMeasure == CmiFalse) UserSetLBLoad();
  //	myRec->getLBDB()->AtLocalBarrier(ldBarrierHandle);

  atsync_iteration++;
  // CkPrintf("[pe %s] atsync_iter %d && predicted period %d state: %d\n",
  //     idx2str(thisIndexMax), atsync_iteration,
  //     myRec->getLBDB()->getPredictedLBPeriod(), local_state);
  double tmp = prev_load;
  prev_load = myRec->getObjTime();
  double current_load = prev_load - tmp;

  if (atsync_iteration != 0) {
    myRec->getLBDB()->AddLoad(atsync_iteration, current_load);
  }

//
//  if (atsync_iteration == 3) {
//    myRec->getLBDB()->AtLocalBarrier(ldBarrierHandle);
//    return;
//  } else {
//    ResumeFromSync();
//    return;
//  }

  if (atsync_iteration < myRec->getLBDB()->getPredictedLBPeriod()) {
    ResumeFromSync();
  } else if (local_state == DECIDED) {
//    CkPrintf("[pe %s] Went to load balance\n", idx2str(thisIndexMax));
    local_state = LOAD_BALANCE;
    local_state = OFF;
    atsync_iteration = -1;
    prev_load = 0.0;
    myRec->getLBDB()->AtLocalBarrier(ldBarrierHandle);
  } else {
//    CkPrintf("[pe %s] Went to pause state\n", idx2str(thisIndexMax));
    local_state = PAUSE;
  }
}

void CkMigratable::ReadyMigrate(CmiBool ready)
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
	DEBL((AA"Element %s resuming from sync\n"AB,idx2str(el->thisIndexMax)));
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    CpvAccess(_currentObj) = el;
#endif
	el->ResumeFromSync();
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    el->mlogData->resumeCount++;
#endif
}
void CkMigratable::setMigratable(int migratable) 
{
	myRec->setMigratable(migratable);
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

/* no load balancer: need dummy implementations to prevent link error */
void CkMigratable::CkAddThreadListeners(CthThread tid, void *msg)
{
}
#endif


/*CkMigratableList*/
CkMigratableList::CkMigratableList() {}
CkMigratableList::~CkMigratableList() {}

void CkMigratableList::setSize(int s) {
	el.resize(s);
}

void CkMigratableList::put(CkMigratable *v,int atIdx) {
#if CMK_ERROR_CHECKING
	if (atIdx>=length())
		CkAbort("Internal array manager error (CkMigrableList::put index out of bounds)");
#endif
	el[atIdx]=v;
}


/************************** Location Records: *********************************/

//---------------- Base type:
void CkLocRec::weAreObsolete(const CkArrayIndex &idx) {}
CkLocRec::~CkLocRec() { }
void CkLocRec::beenReplaced(void)
    {/*Default: ignore replacement*/}

//Return the represented array element; or NULL if there is none
CkMigratable *CkLocRec::lookupElement(CkArrayID aid) {return NULL;}

//Return the last known processor; or -1 if none
int CkLocRec::lookupProcessor(void) {return -1;}


/*----------------- Local:
Matches up the array index with the local index, an
interfaces with the load balancer on behalf of the
represented array elements.
*/
CkLocRec_local::CkLocRec_local(CkLocMgr *mgr,CmiBool fromMigration,
  CmiBool ignoreArrival, const CkArrayIndex &idx_,int localIdx_)
	:CkLocRec(mgr),idx(idx_),localIdx(localIdx_),
	 running(CmiFalse),deletedMarker(NULL)
{
#if CMK_LBDB_ON
	DEBL((AA"Registering element %s with load balancer\n"AB,idx2str(idx)));
	//BIGSIM_OOC DEBUGGING
	//CkPrintf("LocMgr on %d: Registering element %s with load balancer\n", CkMyPe(), idx2str(idx));
	nextPe = -1;
	asyncMigrate = CmiFalse;
	readyMigrate = CmiTrue;
        enable_measure = CmiTrue;
	bounced  = CmiFalse;
	the_lbdb=mgr->getLBDB();
	ldHandle=the_lbdb->RegisterObj(mgr->getOMHandle(),
		idx2LDObjid(idx),(void *)this,1);
	if (fromMigration) {
		DEBL((AA"Element %s migrated in\n"AB,idx2str(idx)));
		if (!ignoreArrival)  {
			the_lbdb->Migrated(ldHandle, CmiTrue);
		  // load balancer should ignore this objects movement
		//  AsyncMigrate(CmiTrue);
		}
	}
#endif
	/*
		FAULT_EVAC
	*/
	asyncEvacuate = CmiTrue;
}
CkLocRec_local::~CkLocRec_local()
{
	if (deletedMarker!=NULL) *deletedMarker=CmiTrue;
	myLocMgr->reclaim(idx,localIdx);
#if CMK_LBDB_ON
	stopTiming();
	DEBL((AA"Unregistering element %s from load balancer\n"AB,idx2str(idx)));
	the_lbdb->UnregisterObj(ldHandle);
#endif
}
void CkLocRec_local::migrateMe(int toPe) //Leaving this processor
{
	//This will pack us up, send us off, and delete us
//	printf("[%d] migrating migrateMe to %d \n",CkMyPe(),toPe);
	myLocMgr->emigrate(this,toPe);
}

void CkLocRec_local::informIdealLBPeriod(int lb_ideal_period) {
  myLocMgr->informLBPeriod(this, lb_ideal_period);
}

#if CMK_LBDB_ON
void CkLocRec_local::startTiming(int ignore_running) {
  	if (!ignore_running) running=CmiTrue;
	DEBL((AA"Start timing for %s at %.3fs {\n"AB,idx2str(idx),CkWallTimer()));
  	if (enable_measure) the_lbdb->ObjectStart(ldHandle);
}
void CkLocRec_local::stopTiming(int ignore_running) {
	DEBL((AA"} Stop timing for %s at %.3fs\n"AB,idx2str(idx),CkWallTimer()));
  	if ((ignore_running || running) && enable_measure) the_lbdb->ObjectStop(ldHandle);
  	if (!ignore_running) running=CmiFalse;
}
void CkLocRec_local::setObjTime(double cputime) {
	the_lbdb->EstObjLoad(ldHandle, cputime);
}
double CkLocRec_local::getObjTime() {
        LBRealType walltime, cputime;
        the_lbdb->GetObjLoad(ldHandle, walltime, cputime);
        return walltime;
}
#endif

void CkLocRec_local::destroy(void) //User called destructor
{
	//Our destructor does all the needed work
	delete this;
}
//Return the represented array element; or NULL if there is none
CkMigratable *CkLocRec_local::lookupElement(CkArrayID aid) {
	return myLocMgr->lookupLocal(localIdx,aid);
}

//Return the last known processor; or -1 if none
int CkLocRec_local::lookupProcessor(void) {
	return CkMyPe();
}

CkLocRec::RecType CkLocRec_local::type(void)
{
	return local;
}

void CkLocRec_local::addedElement(void) 
{
	//Push everything in the half-created queue into the system--
	// anything not ready yet will be put back in.
	while (!halfCreated.isEmpty()) 
		CkArrayManagerDeliver(CkMyPe(),halfCreated.deq());
}

CmiBool CkLocRec_local::isObsolete(int nSprings,const CkArrayIndex &idx_)
{ 
	int len=halfCreated.length();
	if (len!=0) {
		/* This is suspicious-- the halfCreated queue should be extremely
		 transient.  It's possible we just looked at the wrong time, though;
		 so this is only a warning. 
		*/
		CkPrintf("CkLoc WARNING> %d messages still around for uncreated element %s!\n",
			 len,idx2str(idx));
	}
	//A local element never expires
	return CmiFalse;
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

  //DEBS((AA"   Invoking entry %d on element %s\n"AB,epIdx,idx2str(idx)));
	//CmiBool isDeleted=CmiFalse; //Enables us to detect deletion during processing
	//deletedMarker=&isDeleted;
/*#ifndef CMK_OPTIMIZE
	if (msg) {  Tracing: 
		envelope *env=UsrToEnv(msg);
	//	CkPrintf("ckLocation.C beginExecuteDetailed %d %d \n",env->getEvent(),env->getsetArraySrcPe());
		if (_entryTable[epIdx]->traceEnabled)
			_TRACE_BEGIN_EXECUTE_DETAILED(env->getEvent(),
		    		 ForChareMsg,epIdx,env->getsetArraySrcPe(), env->getTotalsize(), idx.getProjectionID(((CkGroupID)env->getsetArrayMgr())).idx);
	}
#endif*/

  return objHandle;
}

void CkMigratable::timingAfterCall(LDObjHandle objHandle,int *objstopped){
  
/*#ifndef CMK_OPTIMIZE
	if (msg) {  Tracing: 
		if (_entryTable[epIdx]->traceEnabled)
			_TRACE_END_EXECUTE();
	}
#endif*/
//#if CMK_LBDB_ON
//        if (!isDeleted) checkBufferedMigration();   // check if should migrate
//#endif
//	if (isDeleted) return CmiFalse;//We were deleted
//	deletedMarker=NULL;
//	return CmiTrue;
	myRec->stopTiming(1);
#if CMK_LBDB_ON
	if (*objstopped) {
		 getLBDB()->ObjectStart(objHandle);
	}
#endif

 return;
}
/****************************************************************************/


CmiBool CkLocRec_local::invokeEntry(CkMigratable *obj,void *msg,
	int epIdx,CmiBool doFree) 
{

	DEBS((AA"   Invoking entry %d on element %s\n"AB,epIdx,idx2str(idx)));
	CmiBool isDeleted=CmiFalse; //Enables us to detect deletion during processing
	deletedMarker=&isDeleted;
	startTiming();


#if CMK_TRACE_ENABLED
	if (msg) { /* Tracing: */
		envelope *env=UsrToEnv(msg);
	//	CkPrintf("ckLocation.C beginExecuteDetailed %d %d \n",env->getEvent(),env->getsetArraySrcPe());
		if (_entryTable[epIdx]->traceEnabled)
			_TRACE_BEGIN_EXECUTE_DETAILED(env->getEvent(),
		    		 ForChareMsg,epIdx,env->getsetArraySrcPe(), env->getTotalsize(), idx.getProjectionID((((CkGroupID)env->getsetArrayMgr())).idx));
	}
#endif

	if (doFree) 
	   CkDeliverMessageFree(epIdx,msg,obj);
	else /* !doFree */
	   CkDeliverMessageReadonly(epIdx,msg,obj);


#if CMK_TRACE_ENABLED
	if (msg) { /* Tracing: */
		if (_entryTable[epIdx]->traceEnabled)
			_TRACE_END_EXECUTE();
	}
#endif
#if CMK_LBDB_ON
        if (!isDeleted) checkBufferedMigration();   // check if should migrate
#endif
	if (isDeleted) return CmiFalse;//We were deleted
	deletedMarker=NULL;
	stopTiming();
	return CmiTrue;
}

CmiBool CkLocRec_local::deliver(CkArrayMessage *msg,CkDeliver_t type,int opts)
{

	if (type==CkDeliver_queue) { /*Send via the message queue */
		if (opts & CK_MSG_KEEP)
			msg = (CkArrayMessage *)CkCopyMsg((void **)&msg);
		CkArrayManagerDeliver(CkMyPe(),msg,opts);
		return CmiTrue;
	}
	else
	{
		CkMigratable *obj=myLocMgr->lookupLocal(localIdx,
			UsrToEnv(msg)->getsetArrayMgr());
		if (obj==NULL) {//That sibling of this object isn't created yet!
			if (opts & CK_MSG_KEEP)
				msg = (CkArrayMessage *)CkCopyMsg((void **)&msg);
			if (msg->array_ifNotThere()!=CkArray_IfNotThere_buffer) {
				return myLocMgr->demandCreateElement(msg,CkMyPe(),type);
			}
			else {
				DEBS((AA"   BUFFERING message for nonexistent element %s!\n"AB,idx2str(this->idx)));
				halfCreated.enq(msg);
				return CmiTrue;
			}
		}
			
		if (msg->array_hops()>1)
			myLocMgr->multiHop(msg);
		CmiBool doFree = (CmiBool)!(opts & CK_MSG_KEEP);
#if CMK_LBDB_ON
		// if there is a running obj being measured, stop it temporarily
		LDObjHandle objHandle;
		int objstopped = 0;
		if (the_lbdb->RunningObject(&objHandle)) {
			objstopped = 1;
			the_lbdb->ObjectStop(objHandle);
		}
#endif
#if CMK_GRID_QUEUE_AVAILABLE
		// retain a pointer to the sending object (needed later)
		CpvAccess(CkGridObject) = obj;
#endif

	CmiBool status = invokeEntry(obj,(void *)msg,msg->array_ep(),doFree);
	
#if CMK_GRID_QUEUE_AVAILABLE
		CpvAccess(CkGridObject) = NULL;
#endif
#if CMK_LBDB_ON
		if (objstopped) the_lbdb->ObjectStart(objHandle);
#endif
		return status;
	}


}

#if CMK_LBDB_ON

void CkLocRec_local::staticAdaptResumeSync(LDObjHandle h, int lb_ideal_period) {
	CkLocRec_local *el=(CkLocRec_local *)LDObjUserData(h);
	DEBL((AA"Load balancer wants to migrate %s to %d\n"AB,idx2str(el->idx),dest));
	el->adaptResumeSync(lb_ideal_period);
}

void CkLocRec_local::adaptResumeSync(int lb_ideal_period) {
  informIdealLBPeriod(lb_ideal_period);
}

void CkLocRec_local::staticMigrate(LDObjHandle h, int dest)
{
	CkLocRec_local *el=(CkLocRec_local *)LDObjUserData(h);
	DEBL((AA"Load balancer wants to migrate %s to %d\n"AB,idx2str(el->idx),dest));
	el->recvMigrate(dest);
}

void CkLocRec_local::recvMigrate(int toPe)
{
	// we are in the mode of delaying actual migration
 	// till readyMigrate()
	if (readyMigrate) { migrateMe(toPe); }
	else nextPe = toPe;
}

void CkLocRec_local::AsyncMigrate(CmiBool use)  
{
        asyncMigrate = use; 
	the_lbdb->UseAsyncMigrate(ldHandle, use);
}

CmiBool CkLocRec_local::checkBufferedMigration()
{
	// we don't migrate in user's code when calling ReadyMigrate(true)
	// we postphone the action to here until we exit from the user code.
	if (readyMigrate && nextPe != -1) {
	    int toPe = nextPe;
	    nextPe = -1;
	    // don't migrate inside the object call
	    migrateMe(toPe);
	    // don't do anything
	    return CmiTrue;
	}
	return CmiFalse;
}

int CkLocRec_local::MigrateToPe()
{
	int pe = nextPe;
	nextPe = -1;
	return pe;
}

void CkLocRec_local::setMigratable(int migratable)
{
	if (migratable)
  	  the_lbdb->Migratable(ldHandle);
	else
  	  the_lbdb->NonMigratable(ldHandle);
}
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void CkLocRec_local::Migrated(){
    the_lbdb->Migrated(ldHandle, CmiTrue);
}
#endif
#endif

/**
 * Represents a deleted array element (and prevents re-use).
 * These are a debugging aid, usable only by uncommenting a line in
 * the element destruction code.
 */
class CkLocRec_dead:public CkLocRec {
public:
	CkLocRec_dead(CkLocMgr *Narr):CkLocRec(Narr) {}
  
	virtual RecType type(void) {return dead;}
  
	virtual CmiBool deliver(CkArrayMessage *msg,CkDeliver_t type,int opts=0) {
		CkPrintf("Dead array element is %s.\n",idx2str(msg->array_index()));
		CkAbort("Send to dead array element!\n");
		return CmiFalse;
	}
	virtual void beenReplaced(void) 
		{CkAbort("Can't re-use dead array element!\n");}
  
	//Return if this element is now obsolete (it isn't)
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx) {return CmiFalse;}	
};

/**
 * This is the abstract superclass of arrayRecs that keep track of their age,
 * and eventually expire. Its kids are remote and buffering.
 */
class CkLocRec_aging:public CkLocRec {
private:
	int lastAccess;//Age when last accessed
protected:
	//Update our access time
	inline void access(void) {
		lastAccess=myLocMgr->getSpringCount();
	}
	//Return if we are "stale"-- we were last accessed a while ago
	CmiBool isStale(void) {
		if (myLocMgr->getSpringCount()-lastAccess>3) return CmiTrue;
		else return CmiFalse;
	}
public:
	CkLocRec_aging(CkLocMgr *Narr):CkLocRec(Narr) {
		lastAccess=myLocMgr->getSpringCount();
	}
	//Return if this element is now obsolete
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx)=0;
	//virtual void pup(PUP::er &p) { CkLocRec::pup(p); p(lastAccess); }
};


/**
 * Represents a remote array element.  This is just a PE number.
 */
class CkLocRec_remote:public CkLocRec_aging {
private:
	int onPe;//The last known Pe for this element
public:
	CkLocRec_remote(CkLocMgr *Narr,int NonPe)
		:CkLocRec_aging(Narr)
		{
			onPe=NonPe;
#if CMK_ERROR_CHECKING
			if (onPe==CkMyPe())
				CkAbort("ERROR!  'remote' array element on this Pe!\n");
#endif
		}
	//Return the last known processor for this element
	int lookupProcessor(void) {
		return onPe;
	}  
	virtual RecType type(void) {return remote;}
  
	//Send a message for this element.
	virtual CmiBool deliver(CkArrayMessage *msg,CkDeliver_t type,int opts=0) {
		/*FAULT_EVAC*/
		int destPE = onPe;
		if((!CmiNodeAlive(onPe) && onPe != allowMessagesOnly)){
//			printf("Delivery failed because process %d is invalid\n",onPe);
			/*
				Send it to its home processor instead
			*/
			const CkArrayIndex &idx=msg->array_index();
			destPE = getNextPE(idx);
		}
		access();//Update our modification date
		msg->array_hops()++;
		DEBS((AA"   Forwarding message for element %s to %d (REMOTE)\n"AB,
		      idx2str(msg->array_index()),destPE));
		if (opts & CK_MSG_KEEP)
			msg = (CkArrayMessage *)CkCopyMsg((void **)&msg);
		CkArrayManagerDeliver(destPE,msg,opts);
		return CmiTrue;
	}
	//Return if this element is now obsolete
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx) {
		if (myLocMgr->isHome(idx)) 
			//Home elements never become obsolete
			// if they did, we couldn't deliver messages to that element.
			return CmiFalse;
		else if (isStale())
			return CmiTrue;//We haven't been used in a long time
		else
			return CmiFalse;//We're fairly recent
	}
	//virtual void pup(PUP::er &p) { CkLocRec_aging::pup(p); p(onPe); }
};


/**
 * Buffers messages until record is replaced in the hash table, 
 * then delivers all messages to the replacing record.  This is 
 * used when a message arrives for a local element that has not
 * yet been created, buffering messages until the new element finally
 * checks in.
 *
 * It's silly to send a message to an element you won't ever create,
 * so this kind of record causes an abort "Stale array manager message!"
 * if it's left undelivered too long.
 */
class CkLocRec_buffering:public CkLocRec_aging {
private:
	CkQ<CkArrayMessage *> buffer;//Buffered messages.
public:
	CkLocRec_buffering(CkLocMgr *Narr):CkLocRec_aging(Narr) {}
	virtual ~CkLocRec_buffering() {
		if (0!=buffer.length()) {
			CkPrintf("[%d] Warning: Messages abandoned in array manager buffer!\n", CkMyPe());
			CkArrayMessage *m;
			while (NULL!=(m=buffer.deq()))  {
				delete m;
			}
		}
	}
  
	virtual RecType type(void) {return buffering;}
  
	//Buffer a message for this element.
	virtual CmiBool deliver(CkArrayMessage *msg,CkDeliver_t type,int opts=0) {
		DEBS((AA" Queued message for %s\n"AB,idx2str(msg->array_index())));
		if (opts & CK_MSG_KEEP)
			msg = (CkArrayMessage *)CkCopyMsg((void **)&msg);
		buffer.enq(msg);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
		envelope *env = UsrToEnv(msg);
		env->sender = CpvAccess(_currentObj)->mlogData->objID;
#endif
		return CmiTrue;
	}
 
	//This is called when this ArrayRec is about to be replaced.
	// We dump all our buffered messages off on the next guy,
	// who should know what to do with them.
	virtual void beenReplaced(void) {
		DEBS((AA" Delivering queued messages:\n"AB));
		CkArrayMessage *m;
		while (NULL!=(m=buffer.deq())) {
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))         
		DEBUG(CmiPrintf("[%d] buffered message being sent\n",CmiMyPe()));
		envelope *env = UsrToEnv(m);
		Chare *oldObj = CpvAccess(_currentObj);
		CpvAccess(_currentObj) =(Chare *) env->sender.getObject();
		env->sender.type = TypeInvalid;
#endif
		DEBS((AA"Sending buffered message to %s\n"AB,idx2str(m->array_index())));
		myLocMgr->deliverViaQueue(m);
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))         
		CpvAccess(_currentObj) = oldObj;
#endif
		}
	}
  
	//Return if this element is now obsolete
	virtual CmiBool isObsolete(int nSprings,const CkArrayIndex &idx) {
		if (isStale() && buffer.length()>0) {
			/*This indicates something is seriously wrong--
			  buffers should be short-lived.*/
			CkPrintf("[%d] WARNING: %d stale array message(s) found!\n",CkMyPe(),buffer.length());
			CkArrayMessage *msg=buffer[0];
			CkPrintf("Addressed to: ");
			CkPrintEntryMethod(msg->array_ep());
			CkPrintf(" index %s\n",idx2str(idx));
			if (myLocMgr->isHome(idx)) 
				CkPrintf("is this an out-of-bounds array index, or was it never created?\n");
			else //Idx is a remote-home index
				CkPrintf("why weren't they forwarded?\n");
			
			// CkAbort("Stale array manager message(s)!\n");
		}
		return CmiFalse;
	}
  
/*  virtual void pup(PUP::er &p) {
    CkLocRec_aging::pup(p);
    CkArray::pupArrayMsgQ(buffer, p);
    }*/
};

/*********************** Spring Cleaning *****************/
/**
 * Used to periodically flush out unused remote element pointers.
 *
 * Cleaning often will free up memory quickly, but slow things
 * down because the cleaning takes time and some not-recently-referenced
 * remote element pointers might be valid and used some time in 
 * the future.
 *
 * Also used to determine if buffered messages have become stale.
 */
inline void CkLocMgr::springCleaning(void)
{
  nSprings++;

  //Poke through the hash table for old ArrayRecs.
  void *objp;
  void *keyp;
  
  CkHashtableIterator *it=hash.iterator();
  CmiImmediateLock(hashImmLock);
  while (NULL!=(objp=it->next(&keyp))) {
    CkLocRec *rec=*(CkLocRec **)objp;
    CkArrayIndex &idx=*(CkArrayIndex *)keyp;
    if (rec->isObsolete(nSprings,idx)) {
      //This record is obsolete-- remove it from the table
      DEBK((AA"Cleaning out old record %s\n"AB,idx2str(idx)));
      hash.remove(*(CkArrayIndex *)&idx);
      delete rec;
      it->seek(-1);//retry this hash slot
    }
  }
  CmiImmediateUnlock(hashImmLock);
  delete it;
}
void CkLocMgr::staticSpringCleaning(void *forWhom,double curWallTime) {
	DEBK((AA"Starting spring cleaning at %.2f\n"AB,CkWallTimer()));
	((CkLocMgr *)forWhom)->springCleaning();
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
    CkArrayIndex &idx=*(CkArrayIndex *)keyp;
    if (rec->type() != CkLocRec::local) {
      //In the case of taking core out of memory (in BigSim's emulation)
      //the meta data in the location manager are not deleted so we need
      //this condition
      if(_BgOutOfCoreFlag!=1){
        hash.remove(*(CkArrayIndex *)&idx);
        delete rec;
        it->seek(-1);//retry this hash slot
      }
    }
    else {
        callMethod((CkLocRec_local*)rec, &CkMigratable::ckDestroy);
        it->seek(-1);//retry this hash slot
    }
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
}
#endif

/*************************** LocMgr: CREATION *****************************/
CkLocMgr::CkLocMgr(CkGroupID mapID_,CkGroupID lbdbID_,CkArrayIndex& numInitial)
	:thisProxy(thisgroup),thislocalproxy(thisgroup,CkMyPe()),
	 hash(17,0.3)
{
	DEBC((AA"Creating new location manager %d\n"AB,thisgroup));
// moved to _CkMigratable_initInfoInit()
//	CkpvInitialize(CkMigratable_initInfo,mig_initInfo);

	managers.init();
	nManagers=0;
  	firstManager=NULL;
	firstFree=localLen=0;
	duringMigration=CmiFalse;
	nSprings=0;
	CcdCallOnConditionKeepOnPE(CcdPERIODIC_1minute,staticSpringCleaning,(void *)this, CkMyPe());

//Register with the map object
	mapID=mapID_;
	map=(CkArrayMap *)CkLocalBranch(mapID);
	if (map==NULL) CkAbort("ERROR!  Local branch of array map is NULL!");
	mapHandle=map->registerArray(numInitial,thisgroup);

//Find and register with the load balancer
	lbdbID = lbdbID_;
	initLB(lbdbID_);
	hashImmLock = CmiCreateImmediateLock();
}

CkLocMgr::CkLocMgr(CkMigrateMessage* m)
	:IrrGroup(m),thisProxy(thisgroup),thislocalproxy(thisgroup,CkMyPe()),hash(17,0.3)
{
	managers.init();
	nManagers=0;
	firstManager=NULL;
	firstFree=localLen=0;
	duringMigration=CmiFalse;
	nSprings=0;
	CcdCallOnConditionKeepOnPE(CcdPERIODIC_1minute,staticSpringCleaning,(void *)this, CkMyPe());
	hashImmLock = CmiCreateImmediateLock();
}

void CkLocMgr::pup(PUP::er &p){
	IrrGroup::pup(p);
	p|mapID;
	p|mapHandle;
	p|lbdbID;
	mapID = _defaultArrayMapID;
	if(p.isUnpacking()){
		thisProxy=thisgroup;
		CProxyElement_CkLocMgr newlocalproxy(thisgroup,CkMyPe());
		thislocalproxy=newlocalproxy;
		//Register with the map object
		map=(CkArrayMap *)CkLocalBranch(mapID);
		if (map==NULL) CkAbort("ERROR!  Local branch of array map is NULL!");
                CkArrayIndex emptyIndex;
		map->registerArray(emptyIndex,thisgroup);
		// _lbdb is the fixed global groupID
		initLB(lbdbID);

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))     
        int count;
        p | count;
        DEBUG(CmiPrintf("[%d] Unpacking Locmgr %d has %d home elements\n",CmiMyPe(),thisgroup.idx,count));
        homeElementCount = count;

        for(int i=0;i<count;i++){
            CkArrayIndex idx;
            int pe;
            idx.pup(p);
            p | pe;
            DEBUG(CmiPrintf("[%d] idx %s is a home element exisiting on pe %d\n",CmiMyPe(),idx2str(idx),pe));
            inform(idx,pe);
            CkLocRec *rec = elementNrec(idx);
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
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))     
        int count=0,count1=0;
        void *objp;
        void *keyp;
        CkHashtableIterator *it = hash.iterator();
      while (NULL!=(objp=it->next(&keyp))) {
      CkLocRec *rec=*(CkLocRec **)objp;
        CkArrayIndex &idx=*(CkArrayIndex *)keyp;
            if(rec->type() != CkLocRec::local){
                if(homePe(idx) == CmiMyPe()){
                    count++;
                }
            }
        }
        p | count;
        DEBUG(CmiPrintf("[%d] Packing Locmgr %d has %d home elements\n",CmiMyPe(),thisgroup.idx,count));

        it = hash.iterator();
      while (NULL!=(objp=it->next(&keyp))) {
      CkLocRec *rec=*(CkLocRec **)objp;
        CkArrayIndex &idx=*(CkArrayIndex *)keyp;
            CkArrayIndex max = idx;
            if(rec->type() != CkLocRec::local){
                if(homePe(idx) == CmiMyPe()){
                    int pe;
                    max.pup(p);
                    pe = rec->lookupProcessor();
                    p | pe;
                    count1++;
                }
            }
        }
        CmiAssert(count == count1);

#endif

	}
}

void _CkLocMgrInit(void) {
  /* Don't trace our deliver method--it does its own tracing */
  CkDisableTracing(CkIndex_CkLocMgr::deliverInline(0));
}

/// Add a new local array manager to our list.
/// Returns a new CkMigratableList for the manager to store his
/// elements in.
CkMigratableList *CkLocMgr::addManager(CkArrayID id,CkArrMgr *mgr)
{
	CK_MAGICNUMBER_CHECK
	DEBC((AA"Adding new array manager\n"AB));
	//Link new manager into list
	ManagerRec *n=new ManagerRec;
	managers.find(id)=n;
	n->next=firstManager;
	n->mgr=mgr;
	n->elts.setSize(localLen);
	nManagers++;
	firstManager=n;
	return &n->elts;
}

/// Return the next unused local element index.
int CkLocMgr::nextFree(void) {
	if (firstFree>=localLen)
	{//Need more space in the local index arrays-- enlarge them
		int oldLen=localLen;
		localLen=localLen*2+8;
		DEBC((AA"Growing the local list from %d to %d...\n"AB,oldLen,localLen));
		for (ManagerRec *m=firstManager;m!=NULL;m=m->next)
			m->elts.setSize(localLen);
		//Update the free list
		freeList.resize(localLen);
		for (int i=oldLen;i<localLen;i++)
			freeList[i]=i+1;
	}
	int localIdx=firstFree;
	if (localIdx==-1) CkAbort("CkLocMgr free list corrupted!");
	firstFree=freeList[localIdx];
	freeList[localIdx]=-1; //Mark as used
	return localIdx;
}

CkLocRec_remote *CkLocMgr::insertRemote(const CkArrayIndex &idx,int nowOnPe)
{
	DEBS((AA"Remote element %s lives on %d\n"AB,idx2str(idx),nowOnPe));
	CkLocRec_remote *rem=new CkLocRec_remote(this,nowOnPe);
	insertRec(rem,idx);
	return rem;
}

//This element now lives on the given Pe
void CkLocMgr::inform(const CkArrayIndex &idx,int nowOnPe)
{
	if (nowOnPe==CkMyPe())
		return; //Never insert a "remote" record pointing here
	CkLocRec *rec=elementNrec(idx);
	if (rec!=NULL && rec->type()==CkLocRec::local){
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        CmiPrintf("[%d]WARNING!!! Element %d:%s is local but is being told it exists on %d\n",CkMyPe(),idx.dimension,idx2str(idx), nowOnPe);
#endif
		return; //Never replace a local element's record!
	}
	insertRemote(idx,nowOnPe);
}

//Tell this element's home processor it now lives "there"
void CkLocMgr::informHome(const CkArrayIndex &idx,int nowOnPe)
{
	int home=homePe(idx);
	if (home!=CkMyPe() && home!=nowOnPe) {
		//Let this element's home Pe know it lives here now
		DEBC((AA"  Telling %s's home %d that it lives on %d.\n"AB,idx2str(idx),home,nowOnPe));
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
        informLocationHome(thisgroup,idx,home,CkMyPe());
#else
		thisProxy[home].updateLocation(idx,nowOnPe);
#endif
	}
}

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
CkLocRec_local *CkLocMgr::createLocal(const CkArrayIndex &idx,
        CmiBool forMigration, CmiBool ignoreArrival,
        CmiBool notifyHome,int dummy)
{
    int localIdx=nextFree();
    DEBC((AA"Adding new record for element %s at local index %d\n"AB,idx2str(idx),localIdx));
    CkLocRec_local *rec=new CkLocRec_local(this,forMigration,ignoreArrival,idx,localIdx);
    if(!dummy){
        insertRec(rec,idx); //Add to global hashtable
    }   
    if (notifyHome) informHome(idx,CkMyPe());
    return rec; 
}
#else
CkLocRec_local *CkLocMgr::createLocal(const CkArrayIndex &idx, 
		CmiBool forMigration, CmiBool ignoreArrival,
		CmiBool notifyHome)
{
	int localIdx=nextFree();
	DEBC((AA"Adding new record for element %s at local index %d\n"AB,idx2str(idx),localIdx));
	CkLocRec_local *rec=new CkLocRec_local(this,forMigration,ignoreArrival,idx,localIdx);
	insertRec(rec,idx); //Add to global hashtable


	if (notifyHome) informHome(idx,CkMyPe());
	return rec;
}
#endif

//Add a new local array element, calling element's constructor
CmiBool CkLocMgr::addElement(CkArrayID id,const CkArrayIndex &idx,
		CkMigratable *elt,int ctorIdx,void *ctorMsg)
{
	CK_MAGICNUMBER_CHECK
	CkLocRec *oldRec=elementNrec(idx);
	CkLocRec_local *rec;
	if (oldRec==NULL||oldRec->type()!=CkLocRec::local) 
	{ //This is the first we've heard of that element-- add new local record
		rec=createLocal(idx,CmiFalse,CmiFalse,CmiTrue);
	} else 
	{ //rec is *already* local-- must not be the first insertion	
		rec=((CkLocRec_local *)oldRec);
		rec->addedElement();
	}
	if (!addElementToRec(rec,managers.find(id),elt,ctorIdx,ctorMsg)) return CmiFalse;
	elt->ckFinishConstruction();
	return CmiTrue;
}

//As above, but shared with the migration code
CmiBool CkLocMgr::addElementToRec(CkLocRec_local *rec,ManagerRec *m,
		CkMigratable *elt,int ctorIdx,void *ctorMsg)
{//Insert the new element into its manager's local list
	int localIdx=rec->getLocalIndex();
	if (m->elts.get(localIdx)!=NULL) CkAbort("Cannot insert array element twice!");
	m->elts.put(elt,localIdx); //Local element table

//Call the element's constructor
	DEBC((AA"Constructing element %s of array\n"AB,idx2str(rec->getIndex())));
	CkMigratable_initInfo &i=CkpvAccess(mig_initInfo);
	i.locRec=rec;
	i.chareType=_entryTable[ctorIdx]->chareIdx;
	if (!rec->invokeEntry(elt,ctorMsg,ctorIdx,CmiTrue)) return CmiFalse;

#if CMK_OUT_OF_CORE
	/* Register new element with out-of-core */
	PUP::sizer p_getSize; elt->pup(p_getSize);
	elt->prefetchObjID=CooRegisterObject(&CkArrayElementPrefetcher,p_getSize.size(),elt);
#endif
	
	return CmiTrue;
}
void CkLocMgr::updateLocation(const CkArrayIndex &idx,int nowOnPe) {
	inform(idx,nowOnPe);
}

/*************************** LocMgr: DELETION *****************************/
/// This index will no longer be used-- delete the associated elements
void CkLocMgr::reclaim(const CkArrayIndex &idx,int localIdx) {
	CK_MAGICNUMBER_CHECK
	DEBC((AA"Destroying element %s (local %d)\n"AB,idx2str(idx),localIdx));
	//Delete, and mark as empty, each array element
	for (ManagerRec *m=firstManager;m!=NULL;m=m->next) {
		delete m->elts.get(localIdx);
		m->elts.empty(localIdx);
	}
	
	removeFromTable(idx);
	
	//Link local index into free list
	freeList[localIdx]=firstFree;
	firstFree=localIdx;
	
		
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
			thisProxy[home].reclaimRemote(idx,CkMyPe());
	/*	//Install a zombie to keep the living from re-using this index.
		insertRecN(new CkLocRec_dead(this),idx); */
	}
}

void CkLocMgr::reclaimRemote(const CkArrayIndex &idx,int deletedOnPe) {
	DEBC((AA"Our element %s died on PE %d\n"AB,idx2str(idx),deletedOnPe));
	CkLocRec *rec=elementNrec(idx);
	if (rec==NULL) return; //We never knew him
	if (rec->type()==CkLocRec::local) return; //He's already been reborn
	removeFromTable(idx);
	delete rec;
}
void CkLocMgr::removeFromTable(const CkArrayIndex &idx) {
#if CMK_ERROR_CHECKING
	//Make sure it's actually in the table before we delete it
	if (NULL==elementNrec(idx))
		CkAbort("CkLocMgr::removeFromTable called on invalid index!");
#endif
        CmiImmediateLock(hashImmLock);
	hash.remove(*(CkArrayIndex *)&idx);
        CmiImmediateUnlock(hashImmLock);
#if CMK_ERROR_CHECKING
	//Make sure it's really gone
	if (NULL!=elementNrec(idx))
		CkAbort("CkLocMgr::removeFromTable called, but element still there!");
#endif
}

/************************** LocMgr: MESSAGING *************************/
/// Deliver message to this element, going via the scheduler if local
/// @return 0 if object local, 1 if not
int CkLocMgr::deliver(CkMessage *m,CkDeliver_t type,int opts) {
	DEBS((AA"deliver \n"AB));
	CK_MAGICNUMBER_CHECK
	CkArrayMessage *msg=(CkArrayMessage *)m;


	const CkArrayIndex &idx=msg->array_index();
	DEBS((AA"deliver %s\n"AB,idx2str(idx)));
	if (type==CkDeliver_queue)
		_TRACE_CREATION_DETAILED(UsrToEnv(m),msg->array_ep());
	CkLocRec *rec=elementNrec(idx);
	if(rec != NULL){
		DEBS((AA"deliver %s of type %d \n"AB,idx2str(idx),rec->type()));
	}else{
		DEBS((AA"deliver %s rec is null\n"AB,idx2str(idx)));
	}
#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))
#if CMK_LBDB_ON
	if (type==CkDeliver_queue) {
		if (!(opts & CK_MSG_LB_NOTRACE) && the_lbdb->CollectingCommStats()) {
		if(rec!=NULL) the_lbdb->Send(myLBHandle,idx2LDObjid(idx),UsrToEnv(msg)->getTotalsize(), rec->lookupProcessor(), 1);
		else /*rec==NULL*/ the_lbdb->Send(myLBHandle,idx2LDObjid(idx),UsrToEnv(msg)->getTotalsize(),homePe(msg->array_index()), 1);
		}
	}
#endif
#endif
#if CMK_GRID_QUEUE_AVAILABLE
	int gridSrcPE;
	int gridSrcCluster;
	int gridDestPE;
	int gridDestCluster;
	CkMigratable *obj;
	ArrayElement *obj2;
	CkGroupID gid;
	int *data;

	obj = (CkMigratable *) CpvAccess(CkGridObject);   // CkGridObject is a pointer to the sending object (retained earlier)
	if (obj != NULL) {
	  obj2 = dynamic_cast<ArrayElement *> (obj);
	  if (obj2 > 0) {
	    // Get the sending object's array gid and indexes.
	    // These are guaranteed to exist due to the succeeding dynamic cast above.
	    gid = obj2->ckGetArrayID ();
	    data = obj2->thisIndexMax.data ();

	    // Get the source PE and destination PE.
	    gridSrcPE = CkMyPe ();
	    if (rec != NULL) {
	      gridDestPE = rec->lookupProcessor ();
	    } else {
	      gridDestPE = homePe (msg->array_index ());
	    }

	    // Get the source cluster and destination cluster.
	    gridSrcCluster = CmiGetCluster (gridSrcPE);
	    gridDestCluster = CmiGetCluster (gridDestPE);

	    // If the Grid queue interval is greater than zero, it means that the more complicated
	    // technique for registering border objects that exceed a specified threshold of
	    // cross-cluster messages within a specified interval (and deregistering border objects
	    // that do not meet this threshold) is used.  Otherwise a much simpler technique is used
	    // where a border object is registered immediately upon sending a single cross-cluster
	    // message (and deregistered when load balancing takes place).
	    if (obj2->grid_queue_interval > 0) {
	      // Increment the sending object's count of all messages.
	      obj2->msg_count += 1;

	      // If the source cluster and destination cluster differ, this is a Grid message.
	      // (Increment the count of all Grid messages.)
	      if (gridSrcCluster != gridDestCluster) {
		obj2->msg_count_grid += 1;
	      }

	      // If the number of messages exceeds the interval, check to see if the object has
	      // sent enough cross-cluster messages to qualify as a border object.
	      if (obj2->msg_count >= obj2->grid_queue_interval) {
		if (obj2->msg_count_grid >= obj2->grid_queue_threshold) {
		  // The object is a border object; if it is not already registered, register it.
		  if (!obj2->border_flag) {
		    CmiGridQueueRegister (gid.idx, obj2->thisIndexMax.nInts, data[0], data[1], data[2]);
		  }
		  obj2->border_flag = 1;
		} else {
		  // The object is not a border object; if it is registered, deregister it.
		  if (obj2->border_flag) {
		    CmiGridQueueDeregister (gid.idx, obj2->thisIndexMax.nInts, data[0], data[1], data[2]);
		  }
		  obj2->border_flag = 0;
		}
		// Reset the counts.
		obj2->msg_count = 0;
		obj2->msg_count_grid = 0;
	      }
	    } else {
	      if (gridSrcCluster != gridDestCluster) {
		CmiGridQueueRegister (gid.idx, obj2->thisIndexMax.nInts, data[0], data[1], data[2]);
	      }
	    }
	  }

	  // Reset the CkGridObject pointer.
	  CpvAccess(CkGridObject) = NULL;
	}
#endif
	/**FAULT_EVAC*/
	if (rec!=NULL){
		CmiBool result = rec->deliver(msg,type,opts);
		// if result is CmiFalse, than rec is not valid anymore, as the object
		// the message was just delivered to has died or migrated out.
		// Therefore rec->type() cannot be invoked!
		if (result==CmiTrue && rec->type()==CkLocRec::local) return 0;
		else return 1;
		/*if(!result){
			//DEBS((AA"deliver %s failed type %d \n"AB,idx2str(idx),rec->type()));
			DEBS((AA"deliver %s failed \n"AB,idx2str(idx)));
			if(rec->type() == CkLocRec::remote){
				if (opts & CK_MSG_KEEP)
					msg = (CkArrayMessage *)CkCopyMsg((void **)&msg);
				deliverUnknown(msg,type);
			}
		}*/
	}else /* rec==NULL*/ {
		if (opts & CK_MSG_KEEP)
			msg = (CkArrayMessage *)CkCopyMsg((void **)&msg);
		deliverUnknown(msg,type,opts);
		return 1;
	}

}

/// This index is not hashed-- somehow figure out what to do.
CmiBool CkLocMgr::deliverUnknown(CkArrayMessage *msg,CkDeliver_t type,int opts)
{
	CK_MAGICNUMBER_CHECK
	const CkArrayIndex &idx=msg->array_index();
	int onPe=homePe(idx);
	if (onPe!=CkMyPe()) 
	{// Forward the message to its home processor
		DEBM((AA"Forwarding message for unknown %s to home %d \n"AB,idx2str(idx),onPe));
		msg->array_hops()++;
		CkArrayManagerDeliver(onPe,msg,opts);
		return CmiTrue;
	}
	else
	{ // We *are* the home processor:
	//Check if the element's array manager has been registered yet:
	  CkArrMgr *mgr=managers.find(UsrToEnv((void *)msg)->getsetArrayMgr())->mgr;
	  if (!mgr) { //No manager yet-- postpone the message (stupidly)
	    if (CkInRestarting()) {
	      // during restarting, this message should be ignored
	      delete msg;
	    }
	    else {
	      CkArrayManagerDeliver(CkMyPe(),msg); 
            }
	  }
	  else { // Has a manager-- must buffer the message
	    DEBC((AA"Adding buffer for unknown element %s\n"AB,idx2str(idx)));
	    CkLocRec *rec=new CkLocRec_buffering(this);
	    insertRecN(rec,idx);
	    rec->deliver(msg,type);
	  
	    if (msg->array_ifNotThere()!=CkArray_IfNotThere_buffer) 
	    { //Demand-create the element:
	      return demandCreateElement(msg,-1,type);
	    }
	  }
	  return CmiTrue;
	}
}

CmiBool CkLocMgr::demandCreateElement(CkArrayMessage *msg,int onPe,CkDeliver_t type)
{
	CK_MAGICNUMBER_CHECK
	const CkArrayIndex &idx=msg->array_index();
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
	DEBC((AA"Demand-creating element %s on pe %d\n"AB,idx2str(idx),onPe));
	CkArrMgr *mgr=managers.find(UsrToEnv((void *)msg)->getsetArrayMgr())->mgr;
	if (!mgr) CkAbort("Tried to demand-create for nonexistent arrMgr");
	return mgr->demandCreateElement(idx,onPe,ctor,type);
}

//This message took several hops to reach us-- fix it
void CkLocMgr::multiHop(CkArrayMessage *msg)
{
	CK_MAGICNUMBER_CHECK
	int srcPe=msg->array_getSrcPe();
	if (srcPe==CkMyPe())
		DEB((AA"Odd routing: local element %s is %d hops away!\n"AB,idx2str(msg),msg->array_hops()));
	else
	{//Send a routing message letting original sender know new element location
		DEBS((AA"Sending update back to %d for element\n"AB,srcPe,idx2str(msg)));
		thisProxy[srcPe].updateLocation(msg->array_index(),CkMyPe());
	}
}

/************************** LocMgr: ITERATOR *************************/
CkLocation::CkLocation(CkLocMgr *mgr_, CkLocRec_local *rec_)
	:mgr(mgr_), rec(rec_) {}
	
const CkArrayIndex &CkLocation::getIndex(void) const {
	return rec->getIndex();
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
    if (rec->type()==CkLocRec::local) {
      CkLocation loc(this,(CkLocRec_local *)rec);
      dest.addLocation(loc);
    }
  }
  CmiImmediateUnlock(hashImmLock);
  delete it;
}




/************************** LocMgr: MIGRATION *************************/
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void CkLocMgr::pupElementsFor(PUP::er &p,CkLocRec_local *rec,
        CkElementCreation_t type, CmiBool create, int dummy)
{
    p.comment("-------- Array Location --------");
    register ManagerRec *m;
    int localIdx=rec->getLocalIndex();
    CkVec<CkMigratable *> dummyElts;

    for (m=firstManager;m!=NULL;m=m->next) {
        int elCType;
        if (!p.isUnpacking())
        { //Need to find the element's existing type
            CkMigratable *elt=m->element(localIdx);
            if (elt) elCType=elt->ckGetChareType();
            else elCType=-1; //Element hasn't been created
        }
        p(elCType);
        if (p.isUnpacking() && elCType!=-1) {
            CkMigratable *elt=m->mgr->allocateMigrated(elCType,rec->getIndex(),type);
            int migCtorIdx=_chareTable[elCType]->getMigCtor();
                if(!dummy){
			if(create)
                    		if (!addElementToRec(rec,m,elt,migCtorIdx,NULL)) return;
 				}else{
                    CkMigratable_initInfo &i=CkpvAccess(mig_initInfo);
                    i.locRec=rec;
                    i.chareType=_entryTable[migCtorIdx]->chareIdx;
                    dummyElts.push_back(elt);
                    if (!rec->invokeEntry(elt,NULL,migCtorIdx,CmiTrue)) return ;
                }
        }
    }
    if(!dummy){
        for (m=firstManager;m!=NULL;m=m->next) {
            CkMigratable *elt=m->element(localIdx);
            if (elt!=NULL)
                {
                       elt->pup(p);
                }
        }
    }else{
            for(int i=0;i<dummyElts.size();i++){
                CkMigratable *elt = dummyElts[i];
                if (elt!=NULL){
            elt->pup(p);
        		}
                delete elt;
            }
			for (ManagerRec *m=firstManager;m!=NULL;m=m->next) {
                m->elts.empty(localIdx);
            }
        freeList[localIdx]=firstFree;
        firstFree=localIdx;
    }
}
#else
void CkLocMgr::pupElementsFor(PUP::er &p,CkLocRec_local *rec,
		CkElementCreation_t type)
{
	p.comment("-------- Array Location --------");
	register ManagerRec *m;
	int localIdx=rec->getLocalIndex();

	//First pup the element types
	// (A separate loop so ckLocal works even in element pup routines)
	for (m=firstManager;m!=NULL;m=m->next) {
		int elCType;
		if (!p.isUnpacking())
		{ //Need to find the element's existing type
			CkMigratable *elt=m->element(localIdx);
			if (elt) elCType=elt->ckGetChareType();
			else elCType=-1; //Element hasn't been created
		}
		p(elCType);
		if (p.isUnpacking() && elCType!=-1) {
			//Create the element
			CkMigratable *elt=m->mgr->allocateMigrated(elCType,rec->getIndex(),type);
			int migCtorIdx=_chareTable[elCType]->getMigCtor();
			//Insert into our tables and call migration constructor
			if (!addElementToRec(rec,m,elt,migCtorIdx,NULL)) return;
		}
	}
	//Next pup the element data
	for (m=firstManager;m!=NULL;m=m->next) {
		CkMigratable *elt=m->element(localIdx);
		if (elt!=NULL)
                {
                        elt->pup(p);
#if CMK_ERROR_CHECKING
                        if (p.isUnpacking()) elt->sanitycheck();
#endif
                }
	}
}
#endif

/// Call this member function on each element of this location:
void CkLocMgr::callMethod(CkLocRec_local *rec,CkMigratable_voidfn_t fn)
{
	int localIdx=rec->getLocalIndex();
	for (ManagerRec *m=firstManager;m!=NULL;m=m->next) {
		CkMigratable *el=m->element(localIdx);
		if (el) (el->* fn)();
	}
}

/// Call this member function on each element of this location:
void CkLocMgr::callMethod(CkLocRec_local *rec,CkMigratable_voidfn_arg_t fn,     void * data)
{
	int localIdx=rec->getLocalIndex();
	for (ManagerRec *m=firstManager;m!=NULL;m=m->next) {
		CkMigratable *el=m->element(localIdx);
		if (el) (el->* fn)(data);
	}
}

/// return a list of migratables in this local record
void CkLocMgr::migratableList(CkLocRec_local *rec, CkVec<CkMigratable *> &list)
{
        register ManagerRec *m;
        int localIdx=rec->getLocalIndex();

        for (m=firstManager;m!=NULL;m=m->next) {
                CkMigratable *elt=m->element(localIdx);
                if (elt) list.push_back(elt);
        }
}

/// Migrate this local element away to another processor.
void CkLocMgr::emigrate(CkLocRec_local *rec,int toPe)
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

#if CMK_OUT_OF_CORE
	int localIdx=rec->getLocalIndex();
	/* Load in any elements that are out-of-core */
	for (ManagerRec *m=firstManager;m!=NULL;m=m->next) {
		CkMigratable *el=m->element(localIdx);
		if (el) if (!el->isInCore) CooBringIn(el->prefetchObjID);
	}
#endif

	//Let all the elements know we're leaving
	callMethod(rec,&CkMigratable::ckAboutToMigrate);
	/*EVAC*/

//First pass: find size of migration message
	int bufSize;
	{
		PUP::sizer p;
		p(nManagers);
		pupElementsFor(p,rec,CkElementCreation_migrate);
		bufSize=p.size(); 
	}

//Allocate and pack into message
	int doubleSize=bufSize/sizeof(double)+1;
	CkArrayElementMigrateMessage *msg = 
		new (doubleSize, 0) CkArrayElementMigrateMessage;
	msg->idx=idx;
	msg->length=bufSize;
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_)) 
    msg->gid = ckGetGroupID();
#endif
#if CMK_LBDB_ON
	msg->ignoreArrival = rec->isAsyncMigrate()?1:0;
#endif
	/*
		FAULT_EVAC
	*/
	msg->bounced = rec->isBounced();
	{
		PUP::toMem p(msg->packData); 
		p.becomeDeleting(); 
		p(nManagers);
		pupElementsFor(p,rec,CkElementCreation_migrate);
		if (p.size()!=bufSize) {
			CkError("ERROR! Array element claimed it was %d bytes to a "
				"sizing PUP::er, but copied %d bytes into the packing PUP::er!\n",
				bufSize,p.size());
			CkAbort("Array element's pup routine has a direction mismatch.\n");
		}
	}

	DEBM((AA"Migrated index size %s to %d \n"AB,idx2str(idx),toPe));	

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
    sendMlogLocation(toPe,UsrToEnv(msg));
#else
	//Send off message and delete old copy
	thisProxy[toPe].immigrate(msg);
#endif

	duringMigration=CmiTrue;
	delete rec; //Removes elements, hashtable entries, local index
	
	
	duringMigration=CmiFalse;
	//The element now lives on another processor-- tell ourselves and its home
	inform(idx,toPe);
#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))    
	informHome(idx,toPe);
#endif
	CK_MAGICNUMBER_CHECK
}

void CkLocMgr::informLBPeriod(CkLocRec_local *rec, int lb_ideal_period) {
	callMethod(rec,&CkMigratable::recvLBPeriod, (void *)&lb_ideal_period);
}

/**
  Migrating array element is arriving on this processor.
*/
void CkLocMgr::immigrate(CkArrayElementMigrateMessage *msg)
{
	const CkArrayIndex &idx=msg->idx;
		
	PUP::fromMem p(msg->packData); 
	
	int nMsgMan;
	p(nMsgMan);
	if (nMsgMan<nManagers)
		CkAbort("Array element arrived from location with fewer managers!\n");
	if (nMsgMan>nManagers) {
		//Some array managers haven't registered yet-- throw it back
		DEBM((AA"Busy-waiting for array registration on migrating %s\n"AB,idx2str(idx)));
		thisProxy[CkMyPe()].immigrate(msg);
		return;
	}

	//Create a record for this element
#if (!defined(_FAULT_MLOG_) && !defined(_FAULT_CAUSAL_))    
	CkLocRec_local *rec=createLocal(idx,CmiTrue,msg->ignoreArrival,CmiFalse /* home told on departure */ );
#else
    CkLocRec_local *rec=createLocal(idx,CmiTrue,CmiTrue,CmiFalse /* home told on departure */ );
#endif
	
	//Create the new elements as we unpack the message
	pupElementsFor(p,rec,CkElementCreation_migrate);
	if (p.size()!=msg->length) {
		CkError("ERROR! Array element claimed it was %d bytes to a"
			"packing PUP::er, but %d bytes in the unpacking PUP::er!\n",
			msg->length,p.size());
		CkError("(I have %d managers; he claims %d managers)\n",
			nManagers,nMsgMan);
		
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
		DEBM((AA"Migrated into failed processor index size %s resent to %d \n"AB,idx2str(idx),newhomePE));	
		CkLocMgr *mgr = rec->getLocMgr();
		int targetPE=getNextPE(idx);
		//set this flag so that load balancer is not informed when
		//this element migrates
		rec->AsyncMigrate(CmiTrue);
		rec->Bounced(CmiTrue);
		mgr->emigrate(rec,targetPE);
		
	}

	delete msg;
}

void CkLocMgr::restore(const CkArrayIndex &idx, PUP::er &p)
{
	//This is in broughtIntoMem during out-of-core emulation in BigSim,
	//informHome should not be called since such information is already
	//immediately updated real migration
#if CMK_ERROR_CHECKING
	if(_BgOutOfCoreFlag!=2)
	    CmiAbort("CkLocMgr::restore should only be used in out-of-core emulation for BigSim and be called when object is brought into memory!\n");
#endif
	CkLocRec_local *rec=createLocal(idx,CmiFalse,CmiFalse,CmiFalse);
	
	//BIGSIM_OOC DEBUGGING
	//CkPrintf("Proc[%d]: Registering element %s with LDB\n", CkMyPe(), idx2str(idx));

	//Create the new elements as we unpack the message
	pupElementsFor(p,rec,CkElementCreation_restore);

	callMethod(rec,&CkMigratable::ckJustRestored);
}


/// Insert and unpack this array element from this checkpoint (e.g., from CkLocation::pup)
#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void CkLocMgr::resume(const CkArrayIndex &idx, PUP::er &p, CmiBool create, int dummy)
{
	CkLocRec_local *rec;
	CkLocRec *recGlobal;	

	if(create){
		rec = createLocal(idx,CmiFalse,CmiFalse,CmiTrue && !dummy /* home doesn't know yet */,dummy );
	}else{
		recGlobal = elementNrec(idx);
		if(recGlobal == NULL) 
			CmiAbort("Local object not found");
		if(recGlobal->type() != CkLocRec::local)
			CmiAbort("Local object not local, :P");
		rec = (CkLocRec_local *)recGlobal;
	}
        
    pupElementsFor(p,rec,CkElementCreation_resume,create,dummy);

    if(!dummy){
        callMethod(rec,&CkMigratable::ckJustMigrated);
    }
}
#else
void CkLocMgr::resume(const CkArrayIndex &idx, PUP::er &p, CmiBool notify)
{
	CkLocRec_local *rec=createLocal(idx,CmiFalse,CmiFalse,notify /* home doesn't know yet */ );

	//Create the new elements as we unpack the message
	pupElementsFor(p,rec,CkElementCreation_resume);

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

//Look up the object with this array index, or return NULL
CkMigratable *CkLocMgr::lookup(const CkArrayIndex &idx,CkArrayID aid) {
	CkLocRec *rec=elementNrec(idx);
	if (rec==NULL) return NULL;
	else return rec->lookupElement(aid);
}
//"Last-known" location (returns a processor number)
int CkLocMgr::lastKnown(const CkArrayIndex &idx) {
	CkLocMgr *vthis=(CkLocMgr *)this;//Cast away "const"
	CkLocRec *rec=vthis->elementNrec(idx);
	int pe=-1;
	if (rec!=NULL) pe=rec->lookupProcessor();
	if (pe==-1) return homePe(idx);
	else{
		/*
			FAULT_EVAC
			if the lastKnownPE is invalid return homePE and delete this record
		*/
		if(!CmiNodeAlive(pe)){
			removeFromTable(idx);
			return homePe(idx);
		}
		return pe;
	}	
}
/// Return true if this array element lives on another processor
bool CkLocMgr::isRemote(const CkArrayIndex &idx,int *onPe) const
{
	CkLocMgr *vthis=(CkLocMgr *)this;//Cast away "const"
	CkLocRec *rec=vthis->elementNrec(idx);
	if (rec==NULL || rec->type()!=CkLocRec::remote) 
		return false; /* not definitely a remote element */
	else /* element is indeed remote */
	{
		*onPe=rec->lookupProcessor();
		return true;
	}
}

static const char *rec2str[]={
    "base (INVALID)",//Base class (invalid type)
    "local",//Array element that lives on this Pe
    "remote",//Array element that lives on some other Pe
    "buffering",//Array element that was just created
    "dead"//Deleted element (for debugging)
};

#if (defined(_FAULT_MLOG_) || defined(_FAULT_CAUSAL_))
void CkLocMgr::setDuringMigration(CmiBool _duringMigration){
    duringMigration = _duringMigration;
}
#endif


//Add given element array record at idx, replacing the existing record
void CkLocMgr::insertRec(CkLocRec *rec,const CkArrayIndex &idx) {
	CkLocRec *old=elementNrec(idx);
	insertRecN(rec,idx);
	if (old!=NULL) {
		DEBC((AA"  replaces old rec(%s) for %s\n"AB,rec2str[old->type()],idx2str(idx)));
		//There was an old element at this location
		if (old->type()==CkLocRec::local && rec->type()==CkLocRec::local) {
		    if (!CkInRestarting()) {    // ok if it is restarting
			CkPrintf("ERROR! Duplicate array index: %s\n",idx2str(idx));
			CkAbort("Duplicate array index used");
		    }
		}
		old->beenReplaced();
		delete old;
	}
}

//Add given record, when there is guarenteed to be no prior record
void CkLocMgr::insertRecN(CkLocRec *rec,const CkArrayIndex &idx) {
	DEBC((AA"  adding new rec(%s) for %s\n"AB,rec2str[rec->type()],idx2str(idx)));
        CmiImmediateLock(hashImmLock);
	hash.put(*(CkArrayIndex *)&idx)=rec;
        CmiImmediateUnlock(hashImmLock);
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
	return hash.getRef(*(CkArrayIndex *)&idx);
#else
//Include an out-of-bounds check if the element isn't found
	CkLocRec *rec=elementNrec(idx);
	if (rec==NULL) abort_out_of_bounds(idx);
	return rec;
#endif
}

//Look up array element in hash table.  Return NULL if not there.
CkLocRec *CkLocMgr::elementNrec(const CkArrayIndex &idx) {
	return hash.get(*(CkArrayIndex *)&idx);
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
void CkLocMgr::initLB(CkGroupID lbdbID_) {}
void CkLocMgr::startInserting(void) {}
void CkLocMgr::doneInserting(void) {}
void CkLocMgr::dummyAtSync(void) {}
#endif


#if CMK_LBDB_ON
void CkLocMgr::initLB(CkGroupID lbdbID_)
{ //Find and register with the load balancer
	the_lbdb = (LBDatabase *)CkLocalBranch(lbdbID_);
	if (the_lbdb == 0)
		CkAbort("LBDatabase not yet created?\n");
	DEBL((AA"Connected to load balancer %p\n"AB,the_lbdb));

	// Register myself as an object manager
	LDOMid myId;
	myId.id = thisgroup;
	LDCallbacks myCallbacks;
	myCallbacks.migrate = (LDMigrateFn)CkLocRec_local::staticMigrate;
	myCallbacks.setStats = NULL;
	myCallbacks.queryEstLoad = NULL;
  myCallbacks.adaptResumeSync =
      (LDAdaptResumeSyncFn)CkLocRec_local::staticAdaptResumeSync;
	myLBHandle = the_lbdb->RegisterOM(myId,this,myCallbacks);

	// Tell the lbdb that I'm registering objects
	the_lbdb->RegisteringObjects(myLBHandle);

	/*Set up the dummy barrier-- the load balancer needs
	  us to call Registering/DoneRegistering during each AtSync,
	  and this is the only way to do so.
	*/
	the_lbdb->AddLocalBarrierReceiver(
		(LDBarrierFn)staticRecvAtSync,(void*)(this));
	dummyBarrierHandle = the_lbdb->AddLocalBarrierClient(
		(LDResumeFn)staticDummyResumeFromSync,(void*)(this));
	dummyAtSync();
}
void CkLocMgr::dummyAtSync(void)
{
	DEBL((AA"dummyAtSync called\n"AB));
	the_lbdb->AtLocalBarrier(dummyBarrierHandle);
}

void CkLocMgr::staticDummyResumeFromSync(void* data)
{      ((CkLocMgr*)data)->dummyResumeFromSync(); }
void CkLocMgr::dummyResumeFromSync()
{
	DEBL((AA"DummyResumeFromSync called\n"AB));
	the_lbdb->DoneRegisteringObjects(myLBHandle);
	dummyAtSync();
}
void CkLocMgr::staticRecvAtSync(void* data)
{      ((CkLocMgr*)data)->recvAtSync(); }
void CkLocMgr::recvAtSync()
{
	DEBL((AA"recvAtSync called\n"AB));
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



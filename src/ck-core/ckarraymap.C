// Definition of ArrayMap base class, as well as some built in maps
#include "ckarraymap.h"
#include "hilbert.h"
#include "TopoManager.h"
#include "partitioning_strategies.h"

//whether to use block mapping in the SMP node level
bool useNodeBlkMapping;

// Group IDs for default maps
CkGroupID _defaultArrayMapID;
CkGroupID _fastArrayMapID;

/*********************** Array Map ******************
Given an array element index, an array map tells us
the index's "home" Pe.  This is the Pe the element will
be created on, and also where messages to this element will
be forwarded by default.
*/

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

int CkArrayMapObj::flattenIndex(const CkArrayOptions& options, const CkArrayIndex& idx) const {
  if (idx.dimension == 1) {
    return idx.data()[0];
  } else if (options.getBounds().dimension == 0) {
    return (idx.hash() + 739) % 1280107;
  }

  CkAssert(idx.dimension == options.getBounds().dimension);

  int flat = idx.data()[0];
  const CkArrayIndex& bounds = options.getBounds();
  if (idx.dimension <= 3) {
    for (int d = 1; d < idx.dimension - 1; d++) {
      flat = flat * bounds.data()[d] + idx.data()[d];
    }
  } else {
    for (int d = 1; d < idx.dimension - 1; d++) {
      flat = flat * ((short int*)bounds.data())[d] + ((short int*)idx.data())[d];
    }
  }

  return flat;
}

void CkArrayMapObj::populateInitial(const CkArrayOptions& options, void* ctorMsg, CkArray* mgr) const {
  // TODO: one way for subclasses to override would be to modify options for only their PEs
  // start end and step, and then call this method.
  CkArrayIndex start = options.getStart();
  CkArrayIndex end = options.getEnd();
  CkArrayIndex step = options.getStep();

  // TODO: Check against bounds
  // TODO: Make sure to not insert if end is <= start
  if (end.nInts == 0 || start == end) {
    mgr->doneInserting();
    CkFreeMsg(ctorMsg);
    return;
  }

  if (start.dimension <= 3) {
    CkArrayIndex i = start;
    bool done = false;
    while (!done) {
      if (CMK_RANK_0(homePe(options, i)) == CkMyPe()) {
        mgr->insertInitial(i, CkCopyMsg(&ctorMsg));
      }
      for (int d = start.dimension - 1; d >= 0; d--) {
        i.data()[d] += step.data()[d];
        if (i.data()[d] < end.data()[d]) {
          break;
        } else if (d > 0) {
          i.data()[d] = start.data()[d];
        } else {
          done = true;
        }
      }
    }
  } else {
    CkArrayIndex i = start;
    bool done = false;
    while (!done) {
      if (CMK_RANK_0(homePe(options, i)) == CkMyPe()) {
        mgr->insertInitial(i, CkCopyMsg(&ctorMsg));
      }
      for (int d = start.dimension - 1; d >= 0; d--) {
        ((short int*)i.data())[d] += ((short int*)step.data())[d];
        if (((short int*)i.data())[d] < ((short int*)end.data())[d]) {
          break;
        } else if (d > 0) {
          ((short int*)i.data())[d] = ((short int*)start.data())[d];
        } else {
          done = true;
        }
      }
    }
  }

  mgr->doneInserting();
  CkFreeMsg(ctorMsg);
}

class RRMapObj : public CkArrayMapObj {
public:
  RRMapObj() {}

  int homePe(const CkArrayOptions& opts, const CkArrayIndex& i) const {
    int flati = flattenIndex(opts, i);
    int home = flati % CkNumPes();
  #if CMK_FAULT_EVAC
    while(!CmiNodeAlive(home) || (home == CkMyPe() && CkpvAccess(startedEvac))) {
      home = (home + 1) % CkNumPes();
    }
  #endif
    return home;
  }
};

class RRMap : public CkArrayMap {
public:
  RRMap() {}
  RRMap(CkMigrateMessage* m) : CkArrayMap(m) {}
  CkArrayMapObj* getMapObj() const { return new RRMapObj(); }
};

/**
 * The default map object -- This does blocked mapping in the general case and
 * calls the round-robin homePe for the dynamic insertion case -- ASB
 */
class DefaultArrayMapObj : public CkArrayMapObj {
protected:
  void computeBlockData(const CkArrayOptions& opts,
      int* totalChares, int* blockSize, int* firstSet, int* remainder) const {
    // TODO: Should we base intial chares on bounds, or initial elems? I could
    // see a desire to have extra elems to be created RR.
    *totalChares = 0;
    if (opts.getBounds().dimension >= 1) *totalChares = 1;
    for (int d = 0; d < opts.getBounds().dimension; d++) {
      if (opts.getBounds().dimension <= 3) {
        *totalChares *= opts.getBounds().data()[d];
      } else {
        *totalChares *= ((short int*)opts.getBounds().data())[d];
      }
    }
    // TODO: What do do in cases of shrink/expand, chkpt/restart etc, where
    // numPEs may change?
    *blockSize = std::max(*totalChares / CkNumPes(), 1);
    *remainder = *totalChares % CkNumPes();
    *firstSet = *remainder * (*blockSize + 1);
  }

public:
  DefaultArrayMapObj() {}

  int homePe(const CkArrayOptions& opts, const CkArrayIndex& i) const {
    int flati = flattenIndex(opts, i);
    int totalChares, blockSize, firstSet, remainder;
    // TODO: This will be removed and stored in member variables when we convert
    // maps from groups to regular objects.
    computeBlockData(opts, &totalChares, &blockSize, &firstSet, &remainder);
    if (flati < firstSet) {
      return flati / (blockSize + 1);
    } else if (flati < totalChares) {
      return ((flati - firstSet) / blockSize) + remainder;
    } else {
      return flati % CkNumPes();
    }
  }
};

class DefaultArrayMap : public RRMap {
public:
  DefaultArrayMap() {}
  DefaultArrayMap(CkMigrateMessage* m) : RRMap(m) {}
  CkArrayMapObj* getMapObj() const { return new DefaultArrayMapObj(); }
};

/**
 *  A fast map for chare arrays which do static insertions and promise NOT
 *  to do late insertions -- ASB
 */
class FastArrayMapObj : public DefaultArrayMapObj {
public:
  FastArrayMapObj() {}

  int homePe(const CkArrayOptions& opts, const CkArrayIndex& i) const {
    int flati = flattenIndex(opts, i);
    int totalChares, blockSize, firstSet, remainder;
    // TODO: This will be removed and stored in member variables when we convert
    // maps from groups to regular objects.
    computeBlockData(opts, &totalChares, &blockSize, &firstSet, &remainder);
    return (flati / blockSize);
  }
};

class FastArrayMap : public DefaultArrayMap {
public:
  FastArrayMap() {}
  FastArrayMap(CkMigrateMessage *m) : DefaultArrayMap(m) {}
  CkArrayMapObj* getMapObj() const { return new FastArrayMapObj(); }
};

#if 0
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
  std::vector<int> procList;
public:
  HilbertArrayMap(void) {
    procList.resize(CkNumPes());
    getHilbertList(procList.data());
    DEBC((AA "Creating HilbertArrayMap\n" AB));
  }

  HilbertArrayMap(CkMigrateMessage *m) : DefaultArrayMap(m){}

  ~HilbertArrayMap() {}

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

  int homePe(int arrayHdl, const CkArrayIndex &i) {
    int flati = 0;
    int myInt;
    int dest;
    if (amaps[arrayHdl]->_nelems.dimension == 0) {
      return RRMap::homePe(arrayHdl, i);
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
  std::vector<int> mapping;

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
        if (fscanf(mapf, "%d %d %d %d", &x, &y, &z, &t) != 4) {
          CkAbort("ReadFileMap> reading from mapfile failed!");
        }
        mapping[i] = tmgr.coordinatesToRank(x, y, z, t);
      }
      fclose(mapf);
    }

    return idx;
  }

  int homePe(int arrayHdl, const CkArrayIndex &i) {
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

// This is currently  here for backwards compatibility
class BlockMap : public DefaultArrayMap {
public:
  BlockMap() {}
  BlockMap(CkMigrateMessage *m) : DefaultArrayMap(m) {}
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
  /*int homePe(int arrayHdl, const CkArrayIndex &i)
  {
     return CLD_ANYWHERE;   // -1
  }*/
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

  std::vector<int> locations;
  int objs_per_block;
  int PE_per_block;

  /// labels for states used when parsing the ConfigurableRRMap from ARGV
  enum ConfigurableRRMapLoadStatus : uint8_t {
    not_loaded,
    loaded_found,
    loaded_not_found
  };

  enum ConfigurableRRMapLoadStatus state;

  ConfigurableRRMapLoader(){
    state = not_loaded;
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
	locations.resize(objs_per_block);
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
   std::vector<int> _map;
 public:
   arrInfo() {}
   arrInfo(const CkArrayIndex& n, int *speeds) : _nelems(n), _map(_nelems.getCombinedCount())
   {
     distrib(speeds);
   }
   ~arrInfo() {}
   int getMap(const CkArrayIndex &i);
   void distrib(int *speeds);
   void pup(PUP::er& p){
     p|_nelems;
     p|_map;
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
  std::vector<double> nspeeds(npes);
  for(i=0;i<npes;i++)
    nspeeds[i] = (double) speeds[i] / total;
  std::vector<int> cp(npes);
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
    std::vector<int> pes(npes);
    for(i=0;i<npes;i++)
      pes[i] = i;
    qsort(pes.data(), npes, sizeof(int), cmp);
    for(i=0;i<nr;i++)
      cp[pes[i]]++;
    delete[] CkpvAccess(rem);
  }
  k = 0;
  for(i=0;i<npes;i++)
  {
    for(j=0;j<cp[i];j++)
      _map[k++] = i;
  }
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
  int s = LBManager::ProcessorSpeed();
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
  int homePe(int arrayHdl, const CkArrayIndex &i)
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

#endif

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

#include "CkArrayMap.def.h"

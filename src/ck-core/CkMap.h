#ifndef _CK_MAP_H
#define _CK_MAP_H

#include "IrrGroup.h"
#include "ckarrayindex.h"

class CkArrayOptions;

/**
\addtogroup CkArray
*/
/*@{*/

#include "ckarrayoptions.h"

/** The "map" is used by the array manager to map an array index to 
 * a home processor number.
 */
class CkArrayMap : public IrrGroup // : public CkGroupReadyCallback
{
public:
  CkArrayMap(void);
  CkArrayMap(CkMigrateMessage *m): IrrGroup(m) {}
  virtual ~CkArrayMap();
  virtual int registerArray(const CkArrayIndex& numElements, CkArrayID aid);
  virtual void unregisterArray(int idx);
  virtual void storeCkArrayOpts(CkArrayOptions options);
  virtual void populateInitial(int arrayHdl,CkArrayOptions& options,void *ctorMsg,CkArray *mgr);
  virtual int procNum(int arrayHdl,const CkArrayIndex &element) =0;
  virtual int homePe(int arrayHdl,const CkArrayIndex &element)
             { return procNum(arrayHdl, element); }

  virtual void pup(PUP::er &p);

  CkArrayOptions storeOpts;
#if CMK_USING_XLC
  std::tr1::unordered_map<int, bool> dynamicIns;
#else
  std::unordered_map<int, bool> dynamicIns;
#endif
};
/*@}*/


class RRMap : public CkArrayMap
{
private:
  CkArrayIndex maxIndex;
  uint64_t products[2*CK_ARRAYINDEX_MAXLEN];
  bool productsInit;

public:
  RRMap(void);
  RRMap(CkMigrateMessage *m);
  void indexInit();
  int procNum(int /*arrayHdl*/, const CkArrayIndex &i);
  void pup(PUP::er& p);
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
  DefaultArrayMap(void);

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
 
  int procNum(int arrayHdl, const CkArrayIndex &i);

  void pup(PUP::er& p);
};


/**
 *  A fast map for chare arrays which do static insertions and promise NOT
 *  to do late insertions -- ASB
 */
class FastArrayMap : public DefaultArrayMap
{
public:
  FastArrayMap(void);

  FastArrayMap(CkMigrateMessage *m) : DefaultArrayMap(m){}

  int registerArray(const CkArrayIndex& numElements, CkArrayID aid)
  {
    int idx;
    idx = DefaultArrayMap::registerArray(numElements, aid);

    return idx;
  }

  int procNum(int arrayHdl, const CkArrayIndex &i);

  void pup(PUP::er& p){
    DefaultArrayMap::pup(p);
  }
};

#endif

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef _ORBLB_H_
#define _ORBLB_H_

#include "CentralLB.h"
#include "OrbLB.decl.h"

void CreateOrbLB();
BaseLB *AllocateOrbLB(void);

class OrbLB : public CBase_OrbLB {
public:/* <- Sun CC demands Partition be public for ComputeLoad to access it. */

  class Partition {
  public:
    int refno;
    double load;			// total load in this set
    int origin[3];			// box coordinates
    int corner[3];
    int  count;				// number of objects in this partition
    int node, mapped;
    CkVec<int>   bkpes;			// background processors
  public:
    Partition(): refno(0), load(0.0), node(-1), mapped(0) {};
  };

private:  
  struct ComputeLoad {
    int id;
    int v[3];
    double load;
    int  refno;
    double  tv;
    Partition * partition;
  };
  
  
  struct VecArray {
    int v;
    int id;
  };
  
  enum {XDIR=0, YDIR, ZDIR};
  
  LDStats* statsData;
  int P;
  ComputeLoad *computeLoad;
  int nObjs;
  VecArray  *(vArray[3]);
  Partition *partitions;
  Partition top_partition;
  int npartition;
  int currentp, refno;
  bool use_udata;
  
  void init();
  void strategy();
  void rec_divide(int, Partition&);
  void setVal(int x, int y, int z);
  int sort_partition(int x, int p, int r);
  void qsort(int x, int p, int r);
  void quicksort(int x);
  void mapPartitionsToNodes();

public:
  double overLoad;

public:
  OrbLB(const CkLBOptions &);
  OrbLB(const CkLBOptions &, bool userdata);
  OrbLB(CkMigrateMessage *m):CBase_OrbLB(m) { init(); }

  void work(LDStats* stats);
  bool QueryBalanceNow(int step);
};


#endif /* _ORBLB_H_ */

/*@}*/

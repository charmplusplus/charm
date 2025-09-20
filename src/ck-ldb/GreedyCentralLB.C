/**
 * \addtogroup CkLdb
*/
/*@{*/

/*
 status:
  * support processor avail bitvector
  * support nonmigratable attrib
      nonmigratable object load is added to its processor's background load
      and the nonmigratable object is not taken in the objData array
*/

#include <algorithm>

#include "charm++.h"


#include "ckgraph.h"
#include "cklists.h"
#include "GreedyCentralLB.h"

using namespace std;

extern int quietModeRequested;

CreateLBFunc_Def(GreedyCentralLB, "always assign the heaviest obj onto lightest loaded processor.")

GreedyCentralLB::GreedyCentralLB(const CkLBOptions &opt): CBase_GreedyCentralLB(opt)
{
  lbname = "GreedyCentralLB";
  if (CkMyPe()==0 && !quietModeRequested)
    CkPrintf("CharmLB> GreedyCentralLB created.\n");
}

bool GreedyCentralLB::QueryBalanceNow(int _step)
{
  //  CkPrintf("[%d] Balancing on step %d\n",CkMyPe(),_step);
  return true;
}

class GreedyCentralLB::ProcLoadGreater {
  public:
    bool operator()(const ProcInfo &p1, const ProcInfo &p2) {
      return (p1.getTotalLoad() > p2.getTotalLoad());
    }
};

class GreedyCentralLB::ObjLoadGreater {
  public:
    bool operator()(const CkVertex &v1, const CkVertex &v2) {
      return (v1.getVertexLoad() > v2.getVertexLoad());
    }
};

void GreedyCentralLB::work(LDStats* stats)
{
  int  obj, objCount, pe;
  int n_pes = stats->nprocs();
  int *map = new int[n_pes];

  std::vector<ProcInfo>  procs;
  for(pe = 0; pe < n_pes; pe++) {
    map[pe] = -1;
    if (stats->procs[pe].available) {
      map[pe] = procs.size();
      procs.push_back(ProcInfo(pe, stats->procs[pe].bg_walltime, 0.0, stats->procs[pe].pe_speed, true));
    }
  }

  // take non migratbale object load as background load
  for (obj = 0; obj < stats->objData.size(); obj++)
  {
      LDObjData &oData = stats->objData[obj];
      if (!oData.migratable)  {
        int pe = stats->from_proc[obj];
        pe = map[pe];
        if (pe==-1)
          CmiAbort("GreedyCentralLB: nonmigratable object on an unavail processor!\n");
        procs[pe].setOverhead(procs[pe].getOverhead() + oData.wallTime);
      }
  }
  delete [] map;

  // Add the overhead to the total load 
  for (pe = 0; pe<procs.size(); pe++) {
    procs[pe].setOverhead(procs[pe].getOverhead() + procs[pe].getOverhead());
  }

  // build object array
  std::vector<CkVertex> objs;

  for(int obj = 0; obj < stats->objData.size(); obj++) {
    LDObjData &oData = stats->objData[obj];
    int pe = stats->from_proc[obj];
    if (!oData.migratable) {
      if (!stats->procs[pe].available) 
        CmiAbort("GreedyCentralLB cannot handle nonmigratable object on an unavial processor!\n");
      continue;
    }
    double load = oData.wallTime * stats->procs[pe].pe_speed;
    objs.push_back(CkVertex(obj, load, stats->objData[obj].migratable, stats->from_proc[obj]));
  }

  // Sort objects by their object IDs
  std::sort(objs.begin(), objs.end(), [&stats](const CkVertex &v1, const CkVertex &v2) {
    return stats->objData[v1.getVertexId()].objID() < stats->objData[v2.getVertexId()].objID();
  });

  int num_remain_objs = objs.size();
  int num_remain_pes = procs.size();
  CkPrintf("[%d] num_remain_objs=%d, num_remain_pes=%d\n", CkMyPe(), num_remain_objs, num_remain_pes);
  int last_obj = 0;

  for (int p=0; p < procs.size(); p++) {
    int nobjs = num_remain_objs / num_remain_pes;
    for (int i=last_obj; i<last_obj + nobjs; i++) {
      stats->to_proc[objs[i].getVertexId()] = procs[p].getProcId();
    }
    last_obj += nobjs;
    num_remain_objs -= nobjs;
    num_remain_pes--;
  }
}

#include "GreedyCentralLB.def.h"

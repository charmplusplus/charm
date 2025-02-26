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

  // max heap of objects
  sort(objs.begin(), objs.end(), GreedyCentralLB::ObjLoadGreater());
  // min heap of processors
  make_heap(procs.begin(), procs.end(), GreedyCentralLB::ProcLoadGreater());

  if (_lb_args.debug()>1) 
    CkPrintf("[%d] In GreedyCentralLB strategy\n",CkMyPe());


    // greedy algorithm
  int nmoves = 0;
  for (obj=0; obj < objs.size(); obj++) {
    ProcInfo p = procs.front();
    pop_heap(procs.begin(), procs.end(), GreedyCentralLB::ProcLoadGreater());
    procs.pop_back();

    // Increment the time of the least loaded processor by the cpuTime of
    // the `heaviest' object
    p.setTotalLoad(p.getTotalLoad() + objs[obj].getVertexLoad() / p.getPeSpeed());

    //Insert object into migration queue if necessary
    const int dest = p.getProcId();
    const int pe   = objs[obj].getCurrentPe();
    const int id   = objs[obj].getVertexId();
    if (dest != pe) {
      stats->to_proc[id] = dest;
      nmoves ++;
      if (_lb_args.debug()>2) 
        CkPrintf("[%d] Obj %d migrating from %d to %d\n", CkMyPe(),objs[obj].getVertexId(),pe,dest);
    }

    //Insert the least loaded processor with load updated back into the heap
    procs.push_back(p);
    push_heap(procs.begin(), procs.end(), GreedyCentralLB::ProcLoadGreater());
  }

  if (_lb_args.debug()>0) 
    CkPrintf("[%d] %d objects migrating.\n", CkMyPe(), nmoves);

  if (_lb_args.debug()>1)  {
    CkPrintf("CharmLB> Min obj: %f  Max obj: %f\n", objs[objs.size()-1].getVertexLoad(), objs[0].getVertexLoad());
    CkPrintf("CharmLB> PE speed:\n");
    for (pe = 0; pe<procs.size(); pe++)
      CkPrintf("%f ", procs[pe].getPeSpeed());
    CkPrintf("\n");
    CkPrintf("CharmLB> PE Load:\n");
    for (pe = 0; pe<procs.size(); pe++)
      CkPrintf("%f (%f)  ", procs[pe].getTotalLoad(), procs[pe].getOverhead());
    CkPrintf("\n");
  }

  if (_lb_args.metaLbOn()) {
    double max_load = 0;
    double avg_load = 0;
    for (pe = 0; pe<procs.size(); pe++) {
      if (procs[pe].getTotalLoad() > max_load) {
        max_load = procs[pe].getTotalLoad();
      }
      avg_load += procs[pe].getTotalLoad();
    }

    stats->after_lb_max = max_load;
    stats->after_lb_avg = avg_load/procs.size();
    stats->is_prev_lb_refine = 0;
    if (_lb_args.debug() > 0)
      CkPrintf("GreedyCentralLB> After lb max load: %lf avg load: %lf\n", max_load, avg_load/procs.size());
  }
}

#include "GreedyCentralLB.def.h"

/*@}*/
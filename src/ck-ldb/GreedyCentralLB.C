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
      return (v1.getCompLoad() > v2.getCompLoad());
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

  // take non migratable object load as background load
  for (obj = 0; obj < stats->objData.size(); obj++)
  {
      LDObjData &oData = stats->objData[obj];
      if (!oData.migratable)  {
        int pe = stats->from_proc[obj];
        pe = map[pe];
        if (pe==-1)
          CmiAbort("GreedyCentralLB: nonmigratable object on an unavail processor!\n");
#if CMK_CUDA
        procs[pe].setOverhead(procs[pe].getOverhead() + std::max(oData.wallTime, oData.gpuTime));
#else
        procs[pe].setOverhead(procs[pe].getOverhead() + oData.wallTime);
#endif
      }
  }
  delete [] map;

  // Add the overhead to the total load
  for (pe = 0; pe<procs.size(); pe++) {
    procs[pe].setTotalLoad(procs[pe].getTotalLoad() + procs[pe].getOverhead());
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
#if CMK_CUDA
    // Use whichever is the bottleneck: CPU wall time or GPU kernel time
    double load = std::max(oData.wallTime, oData.gpuTime) * stats->procs[pe].pe_speed;
    CkPrintf("[%d] GreedyCentralLB obj %d (PE %d): gpuTime=%.6f wallTime=%.6f load=%.6f\n",
             CkMyPe(), obj, pe, oData.gpuTime, oData.wallTime, load);
#else
    double load = oData.wallTime * stats->procs[pe].pe_speed;
#endif
    objs.push_back(CkVertex(obj, load, stats->objData[obj].migratable, stats->from_proc[obj]));
  }

  // max heap of objects (heaviest first)
  sort(objs.begin(), objs.end(), GreedyCentralLB::ObjLoadGreater());
  // min heap of processors (lightest first)
  make_heap(procs.begin(), procs.end(), GreedyCentralLB::ProcLoadGreater());

  if (_lb_args.debug()>1)
    CkPrintf("[%d] In GreedyCentralLB strategy\n",CkMyPe());

  // greedy algorithm: assign heaviest object to lightest processor
  // Use getCompLoad() to avoid the 0.1 floor in getVertexLoad() which
  // destroys load differentiation for fine-grained GPU workloads
  int nmoves = 0;
  for (obj=0; obj < objs.size(); obj++) {
    ProcInfo p = procs.front();
    pop_heap(procs.begin(), procs.end(), GreedyCentralLB::ProcLoadGreater());
    procs.pop_back();

    double obj_load = objs[obj].getCompLoad();
    if (obj_load <= 0.0) obj_load = 1e-6;
    p.setTotalLoad(p.getTotalLoad() + obj_load / p.getPeSpeed());

    //Insert object into migration queue if necessary
    const int dest = p.getProcId();
    const int from_pe = objs[obj].getCurrentPe();
    const int id   = objs[obj].getVertexId();
    if (dest != from_pe) {
      stats->to_proc[id] = dest;
      nmoves ++;
      if (_lb_args.debug()>2)
        CkPrintf("[%d] Obj %d migrating from %d to %d\n", CkMyPe(),id,from_pe,dest);
    }

    //Insert the least loaded processor with load updated back into the heap
    procs.push_back(p);
    push_heap(procs.begin(), procs.end(), GreedyCentralLB::ProcLoadGreater());
  }

  CkPrintf("[%d] GreedyCentralLB: %d objects migrating.\n", CkMyPe(), nmoves);

  if (_lb_args.debug()>1)  {
    CkPrintf("CharmLB> Min obj: %f  Max obj: %f\n", objs[objs.size()-1].getCompLoad(), objs[0].getCompLoad());
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

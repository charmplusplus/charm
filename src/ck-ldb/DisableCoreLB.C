/**
 * \addtogroup CkLdb
*/
/*@{*/

#include <algorithm>

#include "charm++.h"


#include "cklists.h"
#include "DisableCoreLB.h"

using namespace std;

static void lbinit()
{
  LBRegisterBalancer<DisableCoreLB>("DisableCoreLB", "Move objects away from idle cores");
}
DisableCoreLB::DisableCoreLB(const CkLBOptions &opt): CBase_DisableCoreLB(opt)
{
  lbname = "DisableCoreLB";
}

class DisableCoreLB::ProcLoadGreater {
  public:
    bool operator()(const ProcInfo &p1, const ProcInfo &p2) {
      return (p1.getTotalLoad() > p2.getTotalLoad());
    }
};

void DisableCoreLB::work(LDStats* stats)
{
  //Shrink only
  int newPpn = get_active_pes();
//  set_active_redn_pes(get_active_pes());
  CkPrintf("\n newPpn = %d", newPpn);
  int  obj, objCount, pe;
  int n_pes = newPpn;//stats->nprocs();//newPpn;//stats->nprocs();
  int *map = new int[n_pes];

  std::vector<ProcInfo>  procs;//(newPpn);
  for(pe = 0; pe < n_pes; pe++) {
    map[pe] = -1;
    //if (stats->procs[pe].available) 
    {
      map[pe] = procs.size();
      procs.push_back(ProcInfo(pe, stats->procs[pe].bg_walltime, 0.0, stats->procs[pe].pe_speed, true));
    }
  }

  // take non migratbale object load as background load
  for (obj = 0; obj < stats->objData.size(); obj++)
  {
      LDObjData &oData = stats->objData[obj];
#if 0
      if (!oData.migratable)
      {
        int pe = stats->from_proc[obj];
        pe = map[pe];
        if (pe==-1)
          CmiAbort("DisableCoreLB: nonmigratable object on an unavail processor!\n");
        procs[pe].setTotalLoad( procs[pe].getTotalLoad() + oData.wallTime);
      }
#endif
  }
  delete [] map;

  // Add the overhead to the total load 
  for (pe = 0; pe<procs.size(); pe++) {
//    procs[pe].setTotalLoad(procs[pe].getTotalLoad() + procs[pe].getOverhead());
//    procs[pe].setTotalLoad(0.0);
//    procs[pe].setOverhead(0.0);
#if 0 //for stencil remove stale overload when expanding
    if(pe > 0 && procs[pe].getTotalLoad() > procs[0].getTotalLoad()*1.1) {
        procs[pe].setTotalLoad(procs[0].getTotalLoad());
        procs[pe].setOverhead(procs[0].getOverhead());
    }
#endif
//    CkPrintf("\nProc %d load  = %lf", pe, procs[pe].getTotalLoad() + procs[pe].getOverhead());
  }

  procs.resize(get_active_pes());

  // build object array
  std::vector<Vertex> objs;

  for(int obj = 0; obj < stats->objData.size(); obj++) {
    LDObjData &oData = stats->objData[obj];
    int pe = stats->from_proc[obj];
    if (!oData.migratable) {
      if (!stats->procs[pe].available)
        CmiAbort("DisableCoreLB cannot handle nonmigratable object on an unavial processor!\n");
      continue;
    }
    double load = oData.wallTime * stats->procs[pe].pe_speed;
    objs.push_back(Vertex(obj, load, stats->objData[obj].migratable, stats->from_proc[obj]));
  }

  // max heap of objects
  //sort(objs.begin(), objs.end(), DisableCoreLB::ObjLoadGreater());
  // min heap of processors
  make_heap(procs.begin(), procs.end(), DisableCoreLB::ProcLoadGreater());

  if (_lb_args.debug()>1)
    CkPrintf("[%d] In DisableCoreLB strategy [PE-count = %d]\n",CkMyPe(), newPpn);

    // greedy algorithm
  int nmoves = 0;
  for (obj=0; obj < objs.size(); obj++) {
    ProcInfo p = procs.front();
    pop_heap(procs.begin(), procs.end(), DisableCoreLB::ProcLoadGreater());
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
    push_heap(procs.begin(), procs.end(), DisableCoreLB::ProcLoadGreater());
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
}
#include "DisableCoreLB.def.h"

/*@}*/

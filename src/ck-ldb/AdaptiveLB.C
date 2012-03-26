/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "AdaptiveLB.h"
#include "ckgraph.h"

#define alpha 4.0e-6
#define beta 2.67e-9
#define percent_overhead 10

extern LBAllocFn getLBAllocFn(char *lbname);

CreateLBFunc_Def(AdaptiveLB, "Allow multiple strategies to work serially")

AdaptiveLB::AdaptiveLB(const CkLBOptions &opt): CentralLB(opt)
{
  lbname = "AdaptiveLB";
  const char *lbs = theLbdb->loadbalancer(opt.getSeqNo());
  if (CkMyPe() == 0)
    CkPrintf("[%d] AdaptiveLB created with %s\n",CkMyPe(), lbs);

  char *lbcopy = strdup(lbs);
  char *greedyLBString = "GreedyLB";
  char *refineLBString = "RefineLB";
  char *metisLBString = "MetisLB";

  LBAllocFn fn = getLBAllocFn(greedyLBString);
  if (fn == NULL) {
    CkPrintf("LB> Invalid load balancer: %s.\n", greedyLBString);
    CmiAbort("");
  }
  BaseLB *glb = fn();
  greedyLB = (CentralLB*)glb;

  fn = getLBAllocFn(refineLBString);
  if (fn == NULL) {
    CkPrintf("LB> Invalid load balancer: %s.\n", refineLBString);
    CmiAbort("");
  }
  BaseLB *rlb = fn();
  refineLB = (CentralLB*)rlb;

  fn = getLBAllocFn(metisLBString);
  if (fn == NULL) {
    CkPrintf("LB> Invalid load balancer: %s.\n", metisLBString);
    CmiAbort("");
  }
  BaseLB *slb = fn();
  metisLB = (CentralLB*)slb;
}

void AdaptiveLB::work(LDStats* stats)
{

  ProcArray *parr = new ProcArray(stats);
  ObjGraph *ogr = new ObjGraph(stats);
  CkPrintf("Adaptive work\n");

  bool isComm = theLbdb->isStrategyComm();

  // Calculate the load and total messages
  double totalLoad = 0.0;
  long totalMsgs = 0;
  long long totalBytes = 0;
  int vertnbr = ogr->vertices.size();

  /** the object load is normalized to an integer between 0 and 256 */
  for(int i = 0; i < vertnbr; i++) {
    totalLoad += ogr->vertices[i].getVertexLoad();
  }

  for(int i = 0; i < vertnbr; i++) {
    for(int j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
      totalMsgs += ogr->vertices[i].sendToList[j].getNumMsgs();
      totalBytes += ogr->vertices[i].sendToList[j].getNumBytes();
    }
  }
  double commOverhead = (totalMsgs * alpha) + (totalBytes * beta);

  CkPrintf("AdaptiveLB> Total load %E\n", totalLoad);
  CkPrintf("AdaptiveLB> Total Msgs %d\n", totalMsgs);
  CkPrintf("AdaptiveLB> Total Bytes %ld\n", totalBytes);
  CkPrintf("AdaptiveLB> Total Comm Overhead %E Total Load %E\n", commOverhead, totalLoad);

  double after_lb_max;
  double after_lb_avg;
  int is_prev_lb_refine;
  GetPrevLBData(is_prev_lb_refine, after_lb_max, after_lb_avg);

  CkPrintf("AdaptiveLB> Previous LB %d\n", is_prev_lb_refine);

  // Choose the right LB
  //
  // If communication overhead is 10% computation, then choose Scotch LB
  if (isComm || (commOverhead > (totalLoad * percent_overhead / 100))) {
    metisLB->work(stats);
    CkPrintf("---METIS LB\n");
  } else {
    if (is_prev_lb_refine == 1) {
      if (after_lb_max/after_lb_avg < 1.01) {
        refineLB->work(stats);
        CkPrintf("---REFINE LB\n");
      } else {
        greedyLB->work(stats);
        CkPrintf("---GREEDY LB\n");
      }
    } else if (is_prev_lb_refine == -1) {
      greedyLB->work(stats);
      CkPrintf("---GREEDY LB\n");
    } else {
      refineLB->work(stats);
      CkPrintf("---REFINE LB\n");
    }
  }
  UpdateLBDBWithData(stats->is_prev_lb_refine, stats->after_lb_max,
      stats->after_lb_avg);

  delete parr;
  delete ogr;

}

#include "AdaptiveLB.def.h"


/*@}*/

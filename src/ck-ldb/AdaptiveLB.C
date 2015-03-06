/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "AdaptiveLB.h"
#include "ckgraph.h"

#define alpha 4.0e-6
#define beta 2.67e-9
#define percent_overhead 10

extern LBAllocFn getLBAllocFn(const char *lbname);

CreateLBFunc_Def(AdaptiveLB, "Allow multiple strategies to work serially")

AdaptiveLB::AdaptiveLB(const CkLBOptions &opt): CBase_AdaptiveLB(opt)
{
  lbname = "AdaptiveLB";
  const char *lbs = theLbdb->loadbalancer(opt.getSeqNo());
  if (CkMyPe() == 0)
    CkPrintf("[%d] AdaptiveLB created with %s\n",CkMyPe(), lbs);

  char *lbcopy = strdup(lbs);
  const char *greedyLBString = "GreedyLB";
  const char *refineLBString = "RefineLB";
  const char *metisLBString = "MetisLB";
  const char *commRefineLBString = "CommAwareRefineLB";

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

  fn = getLBAllocFn(commRefineLBString);
  if (fn == NULL) {
    CkPrintf("LB> Invalid load balancer: %s.\n", commRefineLBString);
    CmiAbort("");
  }
  BaseLB *crlb = fn();
  commRefineLB = (CentralLB*)crlb;

  if (_lb_args.metaLbOn()) {
    metabalancer = (MetaBalancer*)CkLocalBranch(_metalb);
  } else {
    metabalancer = NULL;
  }
}

void AdaptiveLB::work(LDStats* stats)
{

  ProcArray *parr = new ProcArray(stats);
  ObjGraph *ogr = new ObjGraph(stats);
  CkPrintf("Adaptive work\n");

  //bool isComm = theLbdb->isStrategyComm();

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

  double tmp;
  double refine_max_avg_ratio, lb_max_avg_ratio, greedy_max_avg_ratio;
  int lb_type = -1;
  double comm_ratio, comm_refine_ratio;

  refine_max_avg_ratio = lb_max_avg_ratio = greedy_max_avg_ratio = 1.0;
  comm_ratio = comm_refine_ratio = 1.0;

  if (metabalancer != NULL) {
    metabalancer->GetPrevLBData(lb_type, lb_max_avg_ratio, tmp);
    metabalancer->GetLBDataForLB(0, greedy_max_avg_ratio, tmp);
    metabalancer->GetLBDataForLB(1, refine_max_avg_ratio, tmp);
    metabalancer->GetLBDataForLB(2, tmp, comm_ratio);
    metabalancer->GetLBDataForLB(3, tmp, comm_refine_ratio);
  }

  CkPrintf("AdaptiveLB> Previous LB %d\n", lb_type);

  // Choose the right LB
  //
  // If communication overhead is 10% computation, then choose Scotch LB
  //if (isComm || (commOverhead > (totalLoad * percent_overhead / 100))) {
  if ((commOverhead > (totalLoad * percent_overhead / 100))) {
    if(lb_type == -1) {
      lb_type = 2;
      metisLB->work(stats);
      CkPrintf("---METIS LB\n");
    } else if (comm_refine_ratio <= 1.01) {
      lb_type = 3;
      commRefineLB->work(stats);
      CkPrintf("---CommAwareRefineLB\n");
    } else if (comm_ratio <= 1.01) {
      lb_type = 2;
      metisLB->work(stats);
      CkPrintf("---METIS LB\n");
    } else {
      lb_type = 3;
      commRefineLB->work(stats);
      CkPrintf("---CommAwareRefineLB\n");
    }

  } else {
    if (lb_type == -1) {
      lb_type = 0;
      greedyLB->work(stats);
      CkPrintf("---GREEDY LB\n");
    } else if (refine_max_avg_ratio <= 1.01) {
      lb_type = 1;
      refineLB->work(stats);
      CkPrintf("---REFINE LB\n");
    } else if (greedy_max_avg_ratio <= 1.01) {
      lb_type = 0;
      greedyLB->work(stats);
      CkPrintf("---GREEDY LB\n");
    } else {
      lb_type = 1;
      refineLB->work(stats);
      CkPrintf("---REFINE LB\n");
    }
  }
//  UpdateLBDBWithData(lb_type, stats->after_lb_max, stats->after_lb_avg,
//      stats->local_comm, stats->remote_comm);

  delete parr;
  delete ogr;

}

#include "AdaptiveLB.def.h"


/*@}*/

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
//  char *scotchLBString = "ScotchLB";

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

//  fn = getLBAllocFn(scotchLBString);
//  if (fn == NULL) {
//    CkPrintf("LB> Invalid load balancer: %s.\n", scotchLBString);
//    CmiAbort("");
//  }
//  BaseLB *slb = fn();
//  scotchLB = (CentralLB*)slb;
}

void AdaptiveLB::work(LDStats* stats)
{

  ProcArray *parr = new ProcArray(stats);
  ObjGraph *ogr = new ObjGraph(stats);
  CkPrintf("Adaptive work\n");

  // Calculate the load and total messages
//  double totalLoad = 0.0;
//  long totalMsgs = 0;
//  long long totalBytes = 0;
//  int vertnbr = ogr->vertices.size();
//
//  /** the object load is normalized to an integer between 0 and 256 */
//  for(int i = 0; i < vertnbr; i++) {
//    totalLoad += ogr->vertices[i].getVertexLoad();
//  }
//
//  for(int i = 0; i < vertnbr; i++) {
//    for(int j = 0; j < ogr->vertices[i].sendToList.size(); j++) {
//      totalMsgs += ogr->vertices[i].sendToList[j].getNumMsgs();
//      totalBytes += ogr->vertices[i].sendToList[j].getNumBytes();
//    }
//  }
//  double commOverhead = (totalMsgs * alpha) + (totalBytes * beta);
//
//  CkPrintf("AdaptiveLB> Total load %E\n", totalLoad);
//  CkPrintf("AdaptiveLB> Total Msgs %d\n", totalMsgs);
//  CkPrintf("AdaptiveLB> Total Bytes %ld\n", totalBytes);
//  CkPrintf("AdaptiveLB> Total Comm %E\n", commOverhead);
//
//  // Choose the right LB
//  //
//  // If communication overhead is 10% computation, then choose Scotch LB
//  if (commOverhead > (totalLoad * percent_overhead / 100)) {
//    scotchLB->work(stats);
//  } else {
//    refineLB->work(stats);
//  }
  delete parr;
  delete ogr;

}

#include "AdaptiveLB.def.h"


/*@}*/

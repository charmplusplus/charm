/**************************************************************************
RefineTopoLB: 
This is a topology-aware load balancer.
Author: Tarun Agarwal (tarun)
Date: 04/27/2005
***************************************************************************/
#include <math.h>
#include <stdlib.h>
#include "charm++.h"
#include "cklists.h"
#include "CentralLB.h"

#include "RefineTopoLB.decl.h"

#include "RefineTopoLB.h"

#define alpha PER_MESSAGE_SEND_OVERHEAD_DEFAULT  /*Startup time per message, seconds*/
#define beta PER_BYTE_SEND_OVERHEAD_DEFAULT     /*Long-message time per byte, seconds*/
#define DEG_THRES 0.50
#define EPSILON  -0.001

#define _lb_debug_on 0
#define _lb_debug2_on 1
#define _make_new_grouping_ 1

CreateLBFunc_Def(RefineTopoLB,"TopoLB: Balance objects based on the network topology");


RefineTopoLB::RefineTopoLB(const CkLBOptions &opt) : TopoLB (opt)
{
  lbname = "RefineTopoLB";
  if (CkMyPe () == 0) {
    CkPrintf ("[%d] RefineTopoLB created\n",CkMyPe());
  }
}

CmiBool RefineTopoLB::QueryBalanceNow (int _step)
{
  return CmiTrue;
}

void RefineTopoLB :: work(CentralLB::LDStats *stats,int count)
{
  int i, j;
  if (_lb_args.debug() >= 2) 
  {
    CkPrintf("In TopoLB Strategy...\n");
  }
  
  /****Make sure that there is at least one available processor.***/
  int proc;
  for (proc = 0; proc < count; proc++) 
  {
    if (stats->procs[proc].available)  
      break;
  }
	
  if (proc == count) 
  {
    CmiAbort ("TopoLB: no available processors!");
  }
 
  removeNonMigratable(stats,count);

  if(_lb_debug_on)
  {
    CkPrintf("Num of procs: %d\n",count);
    CkPrintf("Num of objs:  %d\n",stats->n_objs);
  }

  /**************Initialize Topology ****************************/
  LBtopoFn topofn;
  topofn = LBTopoLookup(_lbtopo);
  if (topofn == NULL) 
  {
  	char str[1024];
    CmiPrintf("TopoLB> Fatal error: Unknown topology: %s. Choose from:\n", _lbtopo);
    printoutTopo();
    sprintf(str, "TopoLB> Fatal error: Unknown topology: %s", _lbtopo);
    CmiAbort(str);
  }
  topo = topofn(count);
  /**************************************************************/
  if(_lb_debug_on)
    CkPrintf("before computing partitions...\n");
  
  int *newmap = new int[stats->n_objs];
  if(_make_new_grouping_)
    computePartitions(stats,count,newmap);
  else
  {
    for(int i=0;i<stats->n_objs;i++)
    {
      newmap[i]=stats->from_proc[i];
    }
  }
  /***************** Fill Data Structures *************************/
  if(_lb_debug_on)
    CkPrintf("before allocating dataStructures...\n");
  allocateDataStructures(count);
  if(_lb_debug_on)
    CkPrintf("before initizlizing dataStructures...\n");
  initDataStructures(stats,count,newmap);
  if(_lb_debug_on)
    CkPrintf("After initizlizing dataStructures...\n");
  for(i=0;i<count;i++)
    assign[i]=i;
  if(_lb_debug_on)
    printDataStructures(count, stats->n_objs,newmap);
  /***************** Perform RefineMent *************************/
  bool *swapdone=new bool[count];
  for(i=0;i<count;i++)
    swapdone[i]=false;

    
  //double hbval=getHopBytes(stats,count,stats->from_proc);
  double hbval=getInterMedHopBytes(stats,count,newmap);
 // CkPrintf(" Before Mapping Original   hopBytes : %lf  Avg comm hops: %lf\n", hbval,hbval/total_comm);
  //Perform ith swap
  double totalGain=0;
  for(i=0;i<count;i++)
  {
    //select the cpart which is most communicating and hasn't been moved yet
    int cpart=-1;
    double maxComm=-1;
    for(j=0;j<count;j++)
    {
      if(swapdone[j]) continue;
      if(commUA[j]>maxComm)
      {
        maxComm=commUA[j];
        cpart=j;
      }
    }
    CmiAssert(cpart!=-1);

    //Next find a cpart for swap
    int swapcpart=-1;
    double gainMax=-1;
    double gain=-1;;
    double orig_value=getInterMedHopBytes(stats,count,newmap);
    for(j=0;j<count;j++)
    {
      if(j==cpart)
        continue;

      gain=findSwapGain(j,cpart,count);

      //CkPrintf("%lf : %lf\n",gain,findSwapGain(j,cpart,count));
      if(gain>gainMax && gain>0)
      {
        gainMax=gain;
        swapcpart=j;
      }
    }
    if(swapcpart==-1)
    {
      swapdone[cpart]=true;
      continue;
    }
    totalGain+=gainMax;
    CmiAssert(swapcpart!=-1);
    
    //Actually swap
    int temp=assign[cpart];
    assign[cpart]=assign[swapcpart];
    assign[swapcpart]=temp;
    swapdone[cpart]=true;
  
    //CkPrintf("Gain: %lf  Total_Gain: %lf HopBytes: %lf\n ",gainMax,totalGain,getInterMedHopBytes(stats,count,newmap));
    //CkPrintf(" %lf  getInterMedHopBytes(stats,count,newmap);
    //CkPrintf("Swap# %d:  %d and %d\n",i+1,cpart,swapcpart);
  }
  /******************* Assign mapping and print Stats*********/
  for(i=0;i<stats->n_objs;i++)
  {
    stats->to_proc[i]= assign[newmap[i]];
  }
  if(_lb_debug2_on)
  {
    double hbval=getHopBytes(stats,count,stats->from_proc);
    CkPrintf(" Original   hopBytes : %lf  Avg comm hops: %lf\n", hbval,hbval/total_comm);
  //  hbval=getHopBytes(stats,count,stats->to_proc);
    //CkPrintf(" Resulting  hopBytes : %lf  Avg comm hops: %lf\n", hbval,hbval/total_comm);
    hbval=getInterMedHopBytes(stats,count,newmap);
    CkPrintf("Other Resulting  hopBytes : %lf  Avg comm hops: %lf\n", hbval,hbval/total_comm);
  }
  freeDataStructures(count);
  delete[] newmap;
  delete[] swapdone;
}

double RefineTopoLB::findSwapGain(int cpart1, int cpart2,int count)
{
  double oldvalue=0;
  int proc1=assign[cpart1];
  int proc2=assign[cpart2];
  int proci=-1;

  for(int i=0;i<count;i++)
  {
    proci=assign[i];
    if(i!=cpart1 && i!=cpart2)
    {
      //oldvalue+=comm[cpart1][i]*(dist[proc1][proci]-dist[proc2][proci]);
      //oldvalue+=comm[cpart2][i]*(dist[proc2][proci]-dist[proc1][proci]);
      oldvalue+=(comm[cpart1][i]-comm[cpart2][i])*(dist[proc1][proci]-dist[proc2][proci]);
      
    }
  }
  return oldvalue;
}

double RefineTopoLB::getCpartHopBytes(int cpart, int proc, int count)
{
  double totalHB=0;
  for(int i=0;i<count;i++)
  {
    if(assign[i]==proc)
    {
      totalHB+=comm[cpart][i]*dist[proc][assign[cpart]];
    }
    else
    {
      totalHB+=comm[cpart][i]*dist[proc][assign[i]];
    }
  }
  return totalHB;
}

double RefineTopoLB::getInterMedHopBytes(CentralLB::LDStats *stats,int count, int *newmap)
{
  double totalHB=0;

  for(int i=0;i<count;i++)
  {
    for(int j=0;j<count;j++)
    {
      totalHB+=comm[i][j]*dist[assign[i]][assign[j]];
    }
  }
  return totalHB;
  /*
  CkVec<int> obj_to_proc;
  for(int i=0;i<stats->n_objs;i++)
  {
    obj_to_proc.push_back(assign[newmap[i]]);
  }
  return getHopBytes(stats,count,obj_to_proc);
  */
}

#include "RefineTopoLB.def.h"

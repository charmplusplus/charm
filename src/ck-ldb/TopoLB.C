/**************************************************************************
TopoLB: 
This is a topology-aware load balancer.
 * First groups the objects into $p$ groups.
 * places object-groups onto processors one-by-one
 * Next object group to be placed is the one that gains maximum if placed now. It is placed at the processor that minimizes hop-bytes for it.
 
Author: Tarun Agarwal (tarun)
Date: 04/19/2005
***************************************************************************/

#include <math.h>
#include <stdlib.h>
#include "TopoLB.decl.h"
#include "TopoLB.h"

#define alpha PER_MESSAGE_SEND_OVERHEAD_DEFAULT  /*Startup time per message, seconds*/
#define beta PER_BYTE_SEND_OVERHEAD_DEFAULT     /*Long-message time per byte, seconds*/
#define DEG_THRES 0.50
#define EPSILON  -0.001

#define _lb_debug_on 0
#define _lb_debug2_on 0
#define _make_new_grouping_ 0
#define _FASTER_
#define _DIJKSTRA_LIKE_ 0
#define _INIT_FROM_FILE  


CreateLBFunc_Def(TopoLB,"TopoLB: Balance objects based on the network topology")


TopoLB::TopoLB(const CkLBOptions &opt) : CBase_TopoLB (opt)
{
  lbname = "TopoLB";
  if (CkMyPe () == 0) {
    CkPrintf ("[%d] TopoLB created\n",CkMyPe());
  }
}

bool TopoLB::QueryBalanceNow (int _step)
{
  return true;
}

void TopoLB::freeDataStructures(int count)
{
  for(int i=0;i<count;i++)
  {
    delete[] hopBytes[i];
    delete[] dist[i];
    delete[] comm[i];
  }
  delete[] comm;
  delete[] hopBytes;
  delete[] dist;
  delete[] pfree;
  delete[] cfree;
  delete[] commUA;
  delete[] assign;
}

void TopoLB::allocateDataStructures(int count )
{
  int i;
  //Allocating in separate loop to have somewhat contiguous memory allocation
  hopBytes=new double*[count];
  for(i=0;i<count;i++)
  {
    hopBytes[i]=new double[count+2];
  }

  dist=new double*[count];
  for(i=0;i<count;i++)
  {
    dist[i]=new double[count+1];
  }

  comm=new double*[count];
  for(i=0;i<count;i++)
  {
    comm[i]=new double[count];
  }

  commUA=new double[count];
  pfree=new bool[count];
  cfree=new bool[count];
  assign=new int[count];
}


void TopoLB::computePartitions(CentralLB::LDStats *stats,int count,int *newmap)
{
  int numobjs = stats->n_objs;
	int i, j, m;

  // allocate space for the computing data
  double *objtime = new double[numobjs];
  int *objwt = new int[numobjs];
  int *origmap = new int[numobjs];
  LDObjHandle *handles = new LDObjHandle[numobjs];
  
	for(i=0;i<numobjs;i++) {
    objtime[i] = 0.0;
    objwt[i] = 0;
    origmap[i] = 0;
  }

  for (i=0; i<stats->n_objs; i++) {
    LDObjData &odata = stats->objData[i];
    if (!odata.migratable) 
      CmiAbort("MetisLB doesnot dupport nonmigratable object.\n");
    int frompe = stats->from_proc[i];
    origmap[i] = frompe;
    objtime[i] = odata.wallTime*stats->procs[frompe].pe_speed;
    handles[i] = odata.handle;
  }

  // to convert the weights on vertices to integers
  double max_objtime = objtime[0];
  for(i=0; i<numobjs; i++) {
    if(max_objtime < objtime[i])
      max_objtime = objtime[i];
  }
	int maxobj=0;
	int totalwt=0;
  double ratio = 1000.0/max_objtime;
  for(i=0; i<numobjs; i++) {
      objwt[i] = (int)(objtime[i]*ratio);
			if(maxobj<objwt[i])
				maxobj=objwt[i];
			totalwt+=objwt[i];
  }
	
	//CkPrintf("\nmax obj wt :%d \n totalwt before :%d \n\n",maxobj,totalwt);
	
  int **comm = new int*[numobjs];
  for (i=0; i<numobjs; i++) {
    comm[i] = new int[numobjs];
    for (j=0; j<numobjs; j++)  {
      comm[i][j] = 0;
    }
  }

  const int csz = stats->n_comm;
  for(i=0; i<csz; i++) {
      LDCommData &cdata = stats->commData[i];
      //if(cdata.from_proc() || cdata.receiver.get_type() != LD_OBJ_MSG)
        //continue;
			if(!cdata.from_proc() && cdata.receiver.get_type() == LD_OBJ_MSG){
      	int senderID = stats->getHash(cdata.sender);
      	int recverID = stats->getHash(cdata.receiver.get_destObj());
      	CmiAssert(senderID < numobjs);
      	CmiAssert(recverID < numobjs);
      	//comm[senderID][recverID] += (int)(cdata.messages*alpha + cdata.bytes*beta);
      	//comm[recverID][senderID] += (int)(cdata.messages*alpha + cdata.bytes*beta);
    		//CkPrintf("in compute partitions...%d\n",comm[senderID][recverID]);
				comm[senderID][recverID] += cdata.messages;
      	comm[recverID][senderID] += cdata.messages;

				//Use bytes or messages -- do i include messages for objlist too...??
			}
			else if (cdata.receiver.get_type() == LD_OBJLIST_MSG) {
				//CkPrintf("in objlist..\n");
        int nobjs;
        LDObjKey *objs = cdata.receiver.get_destObjs(nobjs);
        int senderID = stats->getHash(cdata.sender);
        for (j=0; j<nobjs; j++) {
           int recverID = stats->getHash(objs[j]);
           if((senderID == -1)||(recverID == -1))
              if (_lb_args.migObjOnly()) continue;
              else CkAbort("Error in search\n");
           comm[senderID][recverID] += cdata.messages;
           comm[recverID][senderID] += cdata.messages;
        }
			}
		}

// ignore messages sent from an object to itself
  for (i=0; i<numobjs; i++)
    comm[i][i] = 0;

  // construct the graph in CSR format
  int *xadj = new int[numobjs+1];
  int numedges = 0;
  for(i=0;i<numobjs;i++) {
    for(j=0;j<numobjs;j++) {
      if(comm[i][j] != 0)
        numedges++;
    }
  }
  int *adjncy = new int[numedges];
  int *edgewt = new int[numedges];
	int factor = 10;
  xadj[0] = 0;
  int count4all = 0;
  for (i=0; i<numobjs; i++) {
    for (j=0; j<numobjs; j++) { 
      if (comm[i][j] != 0) { 
        adjncy[count4all] = j;
        edgewt[count4all++] = comm[i][j]/factor;
      }
    }
    xadj[i+1] = count4all;
  }

  /*if (_lb_args.debug() >= 2) {
  CkPrintf("Pre-LDB Statistics step %d\n", step());
  printStats(count, numobjs, objtime, comm, origmap);
  }*/

  int wgtflag = 3; // Weights both on vertices and edges
  int numflag = 0; // C Style numbering
  int options[5];
  int edgecut;
  options[0] = 0;

  if (count < 1) {
    CkPrintf("error: Number of Pe less than 1!");
  }
  else if (count == 1) {
   	for(m=0;m<numobjs;m++) 
			newmap[i] = origmap[i];
  }
  else {
		/*
  	if (count > 8)
			METIS_PartGraphKway(&numobjs, xadj, adjncy, objwt, edgewt, 
			    &wgtflag, &numflag, &count, options, 
			    &edgecut, newmap);
	  else
			METIS_PartGraphRecursive(&numobjs, xadj, adjncy, objwt, edgewt, 
				 &wgtflag, &numflag, &count, options, 
				 &edgecut, newmap);
		*/
   //CkPrintf("before calling metis.\n");
	 	METIS_PartGraphRecursive(&numobjs, xadj, adjncy, objwt, edgewt,
                                 &wgtflag, &numflag, &count, options,
                                 &edgecut, newmap);
   	 //CkPrintf("after calling Metis function.\n");
  }
	 
  /*if (_lb_args.debug() >= 2) {
  	CkPrintf("Post-LDB Statistics step %d\n", step());
  	printStats(count, numobjs, objtime, comm, newmap);
  }*/

  for(i=0;i<numobjs;i++)
    delete[] comm[i];
  delete[] comm;
  delete[] objtime;
  delete[] xadj;
  delete[] adjncy;
  delete[] objwt;
  delete[] edgewt;

	//CkPrintf("chking wts on each partition...\n");

	/*int avg=0;
	int *chkwt = new int[count];
	for(i=0;i<count;i++)
		chkwt[i]=0;
	//totalwt=0;
	for(i=0;i<numobjs;i++){
		chkwt[newmap[i]] += objwt[i];
		avg += objwt[i];
	}
	
	for(i=0;i<count;i++)
		CkPrintf("%d -- %d\n",i,chkwt[i]);
	
	//CkPrintf("totalwt after:%d avg is %d\n",avg,avg/count);
*/
  /*if(!sameMapFlag) {
    for(i=0; i<numobjs; i++) {
      if(origmap[i] != newmap[i]) {
				CmiAssert(stats->from_proc[i] == origmap[i]);
				///Dont assign....wait........
				stats->to_proc[i] =  newmap[i];
				if (_lb_args.debug() >= 3)
          CkPrintf("[%d] Obj %d migrating from %d to %d\n", CkMyPe(),i,stats->from_proc[i],stats->to_proc[i]);
      }
    }
  }*/

  delete[] origmap;
}

void TopoLB::initDataStructures(CentralLB::LDStats *stats,int count,int *newmap)
{
  int i;
  //init dist
  if(_lb_debug_on)
    CkPrintf("Before initing dist\n");
 
  topo->get_pairwise_hop_count(dist);
  /*for(i=0;i<count;i++){
   for(int j=0;j<count;j++)
    CkPrintf("%.0lf ",dist[i][j]);
   CkPrintf("\n");
  }*/
  for(i=0;i<count;i++)
  {
    double totaldist=0;
    for(int j=0;j<count;j++)
    {
      //dist[i][j]=topo->get_hop_count(i,j);
      totaldist+=dist[i][j];
    }
    dist[i][count]=totaldist/(count-1);
  }

  //Init comm,commUA from stats
  stats->makeCommHash();
  if(_lb_debug_on)
    CkPrintf("Before initing comm\n");
  for(i=0;i<count;i++)
  {
    for(int j=0;j<count;j++)
    {
      comm[i][j]=0;
    }
    commUA[i]=0;
  }
  bool *multicastAdded=new bool[count];
  for(i=0;i<stats->n_comm;i++)
  {
    LDCommData &cdata=stats->commData[i];
    if(!cdata.from_proc() && cdata.receiver.get_type() ==LD_OBJ_MSG)
    {
      int sender=stats->getHash(cdata.sender);
      int receiver=stats->getHash(cdata.receiver.get_destObj());

      CmiAssert(sender<stats->n_objs);
      CmiAssert(receiver<stats->n_objs);

      if(newmap[sender]==newmap[receiver])
        continue;

      int send_part=newmap[sender];
      int recv_part=newmap[receiver];
      comm[send_part][recv_part]+=cdata.bytes;
      comm[recv_part][send_part]+=cdata.bytes;

      commUA[send_part]+=cdata.bytes;
      commUA[recv_part]+=cdata.bytes;
    }
    if(!cdata.from_proc() && cdata.receiver.get_type()==LD_OBJLIST_MSG)
    {
      int nobjs=0;
      LDObjKey *receivers=cdata.receiver.get_destObjs(nobjs);
      int sender=stats->getHash(cdata.sender);
      int send_part=newmap[sender];
      
      CmiAssert(sender<stats->n_objs);

      for(int i=0;i<count;i++)
        multicastAdded[i]=false;
      multicastAdded[send_part]=true;

      for(int k=0;k<nobjs;k++)
      {
        int receiver=stats->getHash(receivers[k]);
        CmiAssert ( receiver < stats->n_objs);

        int recv_part=newmap[receiver];
        if(!multicastAdded[recv_part])
        {
          comm[send_part][recv_part]+=cdata.bytes;
          comm[recv_part][send_part]+=cdata.bytes;
      
          commUA[send_part]+=cdata.bytes;
          commUA[recv_part]+=cdata.bytes;

          multicastAdded[recv_part]=true;
        }
      }
    }
  }
  /*****FIXED A MEMORY LEAK******/
  delete[] multicastAdded;

  /***************************/
  //Just a test
  /*
  for(int i=0;i<count;i++)
    commUA[0];
  for(int i=0;i<count;i++)
  {
    for(int j=0;j<i;j++)
    {
      comm[j][i]=comm[i][j]= rand()%100;
      commUA[i]+=comm[i][j];
      commUA[j]+=comm[i][j];
    }
  }
  */

  /******Avg degree test *******/
  total_comm=0;
  if(_lb_debug2_on)
  {
    int avg_degree=0;
    for(int i=0;i<count;i++)
    {
      double comm_i_total=0;
      for(int j=0;j<count;j++)
      {
        avg_degree+=(comm[i][j]>0);
        total_comm+=comm[i][j];
        comm_i_total+=comm[i][j];
      }
    }
     // CkPrintf("Avg degree (%d nodes) : %d\n",count,avg_degree/count);
  }
  /***************************/

  //Init hopBytes
  //hopBytes[i][j]=hopBytes if partition i is placed at proc j
  if(_lb_debug_on)
    CkPrintf("Before initing hopBytes\n");
  for(i=0;i<count;i++)
  {
    int hbminIndex=0;
    double hbtotal=0;
    for(int j=0;j<count;j++)
    {
      //Initialize by (total comm of i)*(avg dist from j);
      if(_DIJKSTRA_LIKE_)
        hopBytes[i][j]=0;
      else
        hopBytes[i][j]=commUA[i]*dist[j][count];

      hbtotal+=hopBytes[i][j];
      if(hopBytes[i][hbminIndex]>hopBytes[i][j])
        hbminIndex=j;
    }
    hopBytes[i][count]=hbminIndex;
    hopBytes[i][count+1]=hbtotal/count;
  }
  
  //Init pfree, cfree, assign
  if(_lb_debug_on)
    CkPrintf("Before initing pfree cfree assign\n");
  for(i=0;i<count;i++)
  {
    pfree[i]=true;
    cfree[i]=true;
    assign[i]=-1;
  }

}

void randomAssign(int count, int *perm)
{
  int i;
  for(i=0;i<count;i++)
    perm[i]=i;
  for(i=0;i<count;i++)
  {
    int randpos=i+rand()%(count-i);
    int temp=perm[i];
    perm[i]=perm[randpos];
    perm[randpos]=temp;
  }
}

void TopoLB::performMapping(int *newmap, int count)
{
  int i,j;
  if(_lb_debug_on)
    CkPrintf("before performing mapping...\n");

  double *distnew=new double[count];
  for(i=0;i<count;i++)
  {
    //Assume i-1 partitions placed, 
    //select and place the ith partition

    //select the (c,p) pair
    int part_index=-1;
    int proc_index=-1;


    double gainMax=-1;
    for(j=0;j<count;j++)
    {
      if(!cfree[j])
        continue;
      
      int hb_j_minIndex=(int)hopBytes[j][count];
      //CmiAssert(pfree[hb_j_minIndex]);
      double gain=hopBytes[j][count+1]-hopBytes[j][hb_j_minIndex];

      //CmiAssert(gain>=0);
      //CmiAssert(gain>=EPSILON);
      
      if(_lb_debug_on)
        CkPrintf("Gain is : %lf\n",gain);

      if(gain>gainMax)
      {
        part_index=j;
        proc_index=hb_j_minIndex;
        gainMax=gain;
      }
    }
    if(_lb_debug_on)
        CkPrintf("GainMax [%d] is : %lf\n",i,gainMax);

    CmiAssert(part_index!=-1);
    CmiAssert(proc_index!=-1);

    //Assign the selection
    CmiAssert(assign[part_index]==-1);
    CmiAssert(pfree[proc_index]);
    assign[part_index]=proc_index;
    
    /*
    if(_lb_debug2_on)
    {
      if(i%100 ==0)
        CkPrintf("Assigned %d procs \n",i+1);
    }
    */
    
    
    /*****Update Data Structures******************/


#ifndef _FASTER_
    cfree[part_index]=false;
    pfree[proc_index]=false;

    //Dont need to update other data structures if this is last assignment
    if(i == count-1)
      continue;

    int procs_left=count-i-1;

    //Update hopBytes
    
    for(j=0;j<count;j++)
    {
      if(procs_left>1)
        distnew[j]=(dist[j][count]*procs_left -dist[j][proc_index]) / (procs_left-1);
      else
        distnew[j]=0;
    }
    for(int cpart=0;cpart<count;cpart++)
    {
      if(!cfree[cpart]) //No need to update for assigned partitions
        continue;    

      if(commUA[cpart]==0 && hopBytes[cpart][count]!=proc_index)
      {
        if(procs_left>0)
          hopBytes[cpart][count+1]=(hopBytes[cpart][count+1]*(procs_left+1) - hopBytes[cpart][proc_index])/(procs_left);
        continue;
      }

      //double hbmin=INFTY;
      int h_min_index=-1;
      double h_min=-1;
      double h_total=0;

      double c1=commUA[cpart];
      double c2=comm[cpart][part_index];

      double h_updated=0;

      for(int proc=0;proc<count;proc++)
      {
        if(!pfree[proc]) // No need to update for assigned procs
          continue;

        hopBytes[cpart][proc]+=(c1-c2)*distnew[proc]+c2*dist[proc][proc_index]-c1*dist[proc][count];
        h_updated=hopBytes[cpart][proc];
        
        //CmiAssert(hopBytes[cpart][proc] >= EPSILON);

        h_total+=h_updated;
        if( h_updated<h_min || h_min_index==-1 )
        {
          h_min=h_updated;
          h_min_index=proc;
        }
      }
      //CmiAssert(h_min_index!=-1);
      hopBytes[cpart][count]=h_min_index;
      hopBytes[cpart][count+1]=h_total/(procs_left);
    }

    // d[j][count] is the average dist of proc j to unassigned procs
    // Also update commUA[j]
    for(j=0;j<count;j++)
    {
      if(procs_left>1)
        dist[j][count]=(dist[j][count]*procs_left -dist[j][proc_index]) / (procs_left-1);
      else
        dist[j][count]=0;
      commUA[j]-=comm[j][part_index];
    }

#else
    cfree[part_index]=false;
    pfree[proc_index]=false;

    //Dont need to update other data structures if this is last assignment
    if(i == count-1)
      continue;

    int procs_left=count-i-1;

    //Update hopBytes
    for(int cpart=0;cpart<count;cpart++)
    {
      //No need to update for assigned partitions
      if(!cfree[cpart]) 
        continue;    

      //If cpart doesn't communicate with part_index, just need to update minimum and average hopBytes attainable for cpart
      if(comm[cpart][part_index]==0) 
      {
        //change average
        if(procs_left>0)
          hopBytes[cpart][count+1]=(hopBytes[cpart][count+1]*(procs_left+1) - hopBytes[cpart][proc_index])/(procs_left);

        //change minimum if needed
        if(hopBytes[cpart][count]==proc_index)
        {
          int c_min=-1;
          double c_min_val=-1;
          for(int l=0;l<count;l++)
          {
            if(!pfree[l])
              continue;
            
            if(c_min==-1 || hopBytes[cpart][l]<c_min_val)
            {
              c_min_val=hopBytes[cpart][l];
              c_min=l;
            }
          }
          CmiAssert(c_min!=-1);
          hopBytes[cpart][count]=c_min;
        }
        continue;
      }

      // If reached here cpart communicates with the partition assigned in this step( part_index)
      int h_min_index=-1;
      double h_min=-1;
      double h_total=0;

      //double c1=commUA[cpart];
      double c2=comm[cpart][part_index];

      double h_updated=0;

      for(int proc=0;proc<count;proc++)
      {
        if(!pfree[proc]) // No need to update for assigned procs
          continue;

        if(_DIJKSTRA_LIKE_)
          hopBytes[cpart][proc]+=c2*(dist[proc][proc_index]);
        else
          hopBytes[cpart][proc]+=c2*(dist[proc][proc_index]-dist[proc][count]);
        h_updated=hopBytes[cpart][proc];
        
        //CmiAssert(h_updated >= EPSILON);

        h_total+=h_updated;
        if(h_updated<h_min || h_min_index==-1 )
        {
          h_min=h_updated;
          h_min_index=proc;
        }
      }
      CmiAssert(h_min_index!=-1);
      CmiAssert(pfree[h_min_index]);
      hopBytes[cpart][count]=h_min_index;
      hopBytes[cpart][count+1]=h_total/(procs_left);
    }

#endif
    
  }
 
  /****FIXED a memory leak******/
  delete [] distnew;
  /******************  Fill out final composition Mapping **************************/

}

void TopoLB :: work(LDStats *stats)
{
  int i;
  int n_pes = stats->nprocs();

  if (_lb_args.debug() >= 2) {
    CkPrintf("In TopoLB Strategy...\n");
  }
  
  /****Make sure that there is at least one available processor.***/
  int proc;
  for (proc = 0; proc < n_pes; proc++) {
    if (stats->procs[proc].available)  
      break;
  }
	
  if (proc == n_pes) {
    CmiAbort ("TopoLB: no available processors!");
  }
 
  removeNonMigratable(stats, n_pes);

  if(_lb_debug_on)
  {
    CkPrintf("Num of procs: %d\n", n_pes);
    CkPrintf("Num of objs:  %d\n",stats->n_objs);
  }

  /**************Initialize Topology ****************************/
  LBtopoFn topofn;
  char *lbcopy = strdup(_lbtopo);
  char *ptr = strchr(lbcopy, ':');
  if (ptr!=NULL)
    ptr = strtok(lbcopy, ":");
	else
		ptr=lbcopy;

  topofn = LBTopoLookup(ptr);
  if (topofn == NULL) 
  {
  	char str[1024];
    CmiPrintf("TopoLB> Fatal error: Unknown topology: %s. Choose from:\n", _lbtopo);
    printoutTopo();
    sprintf(str, "TopoLB> Fatal error: Unknown topology: %s", _lbtopo);
    CmiAbort(str);
  }
  topo = topofn(n_pes);

  /*********** Compute Partitions *********************************/
  if(_lb_debug_on)
    CkPrintf("before computing partitions...\n");
  
  int *newmap = new int[stats->n_objs];
  if(_make_new_grouping_)
    computePartitions(stats, n_pes, newmap);
  else
  {
    for(i=0;i<stats->n_objs;i++)
    {
      newmap[i]=stats->from_proc[i];
    }
  }
  /***************** Fill Data Structures *************************/
  if(_lb_debug_on)
    CkPrintf("before allocating dataStructures...\n");

  allocateDataStructures(n_pes);

  if(_lb_debug_on)
    CkPrintf("before initizlizing dataStructures...\n");

  initDataStructures(stats, n_pes, newmap);

  if(_lb_debug_on)
    printDataStructures(n_pes, stats->n_objs, newmap);
  
  /****************** Perform Mapping *****************************/

  if(_lb_debug_on)
    CkPrintf("before performing mapping...\n");
  
  performMapping(newmap, n_pes);

  for(i=0;i<stats->n_objs;i++)
  {
    stats->to_proc[i]= assign[newmap[i]];
  }

  if(_lb_debug2_on)
  {
    double hbval1=getHopBytesNew(NULL, n_pes);
    CkPrintf("\n");
    CkPrintf(" Original hopBytes : %.1lf \n", hbval1);
    CkPrintf(" Original Avg hops : %.1lf \n", hbval1/total_comm);
    double hbval2=getHopBytesNew(assign, n_pes);
    //CkPrintf(" Resulting hopBytes : %.1lf \n", hbval2);
    //CkPrintf(" Resulting Avg hops : %.1lf \n", hbval2/total_comm);
    //CkPrintf(" Percentage gain %.2lf\n",(hbval1-hbval2)*100/hbval1);
    //CkPrintf("\n");
    
    double total_hb_rand=0;
    int nTrials=10;
    for(int t=0;t<nTrials;t++)
    {
      randomAssign(n_pes, assign);
      double hbval3=getHopBytesNew(assign, n_pes);
      total_hb_rand+=hbval3;
     //CkPrintf(" Random hopBytes : %.1lf \n", hbval3);
     //CkPrintf(" Random Avg hops : %.1lf \n", hbval3/total_comm);
    }
    //CkPrintf("\n");
    double hbval4=total_hb_rand/nTrials;
    CkPrintf(" Average Random hopBytes : %.1lf \n", hbval4);
    CkPrintf(" Average Random Avg hops : %.1lf \n", hbval4/total_comm);
    CkPrintf(" Resulting hopBytes : %.1lf \n", hbval2);
    CkPrintf(" Resulting Avg hops : %.1lf \n", hbval2/total_comm);
    CkPrintf(" Percentage gain(TopoLB vs random) %.2lf\n",(hbval4-hbval2)*100/hbval4);
    CkPrintf("\n");
  }
  freeDataStructures(n_pes);
  delete[] newmap;
  return;
}


void TopoLB::printDataStructures(int count,int n_objs,int *newmap)
{
  int i;
  /*
  CkPrintf("Partition Results : \n");
  for(int i=0;i<n_objs;i++)
  {
    CkPrintf("map[%d] = %d\n",i,newmap[i]);
  }
  */

  CkPrintf("Dist : \n");
  for(i=0;i<count;i++)
  {
    for(int j=0;j<count+1;j++)
    {
      CkPrintf("%lf ",dist[i][j]);
    }
    CkPrintf("\n");
  }
  CkPrintf("HopBytes: \n");
  for(i=0;i<count;i++)
  {
    for(int j=0;j<count+2;j++)
    {
      CkPrintf("%lf ",hopBytes[i][j]);
    }
    CkPrintf("\n");
  }
}

double TopoLB::getHopBytesNew(int *assign_map, int count)
{
  double totalHB=0;
  int i,j;
  if(assign_map==NULL)
  {
    for(i=0;i<count;i++)
      for(j=0;j<count;j++)
        totalHB+=dist[i][j]*comm[i][j];
  }
  else
  {
    for(i=0;i<count;i++)
      for(j=0;j<count;j++)
        totalHB+=dist[assign_map[i]][assign_map[j]]*comm[i][j];
  }
  return totalHB;
}

double TopoLB::getHopBytes(CentralLB::LDStats *stats,int count,CkVec<int>map)
{
  int i;
  double **comm1=new double*[count];
  for(i=0;i<count;i++)
    comm1[i]=new double[count];

  for(i=0;i<count;i++)
  {
    for(int j=0;j<count;j++)
    {
      comm1[i][j]=0;
    }
  }

  bool *multicastAdded=new bool[count];
  for(i=0;i<stats->n_comm;i++)
  {
    LDCommData &cdata=stats->commData[i];
    if(!cdata.from_proc() && cdata.receiver.get_type() ==LD_OBJ_MSG)
    {
      int sender=stats->getHash(cdata.sender);
      int receiver=stats->getHash(cdata.receiver.get_destObj());

      CmiAssert(sender<stats->n_objs);
      CmiAssert(receiver<stats->n_objs);

      if(map[sender]==map[receiver])
        continue;

      int send_part=map[sender];
      int recv_part=map[receiver];
      comm1[send_part][recv_part]+=cdata.bytes;
      comm1[recv_part][send_part]+=cdata.bytes;
    }
    if(!cdata.from_proc() && cdata.receiver.get_type()==LD_OBJLIST_MSG)
    {
      int nobjs=0;
      LDObjKey *receivers=cdata.receiver.get_destObjs(nobjs);
      int sender=stats->getHash(cdata.sender);
      int send_part=map[sender];
      
      CmiAssert(sender<stats->n_objs);

      for(i=0;i<count;i++)
        multicastAdded[i]=false;
      multicastAdded[send_part]=true;

      for(int k=0;k<nobjs;k++)
      {
        int receiver=stats->getHash(receivers[k]);
        //CmiAssert ( (int)(receivers[k])< stats->n_objs);
        CmiAssert ( receiver < stats->n_objs);

        int recv_part=map[receiver];
        if(!multicastAdded[recv_part])
        {
          comm1[send_part][recv_part]+=cdata.bytes;
          comm1[recv_part][send_part]+=cdata.bytes;
      
          multicastAdded[recv_part]=true;
        }
      }
    }
  }
  delete[] multicastAdded;

  double totalHB=0;

  for(i=0;i<count;i++)
  {
    for(int j=0;j<count;j++)
    {
      totalHB+=dist[i][j]*comm1[i][j];
    }
  }
  for(i=0;i<count;i++)
    delete[] comm1[i];
  delete[] comm1;

  return totalHB;
}

#include "TopoLB.def.h"

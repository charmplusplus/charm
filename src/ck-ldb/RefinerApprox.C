#include "elements.h"
#include "ckheap.h"
#include "RefinerApprox.h"

int _lb_debug=0;

void RefinerApprox::create(int count, CentralLB::LDStats* stats, int* procs)
{
  int i;
  // now numComputes is all the computes: both migratable and nonmigratable.
  // afterwards, nonmigratable computes will be taken off

  numAvail = 0;
  for(i=0; i < P; i++) 
  {
    processors[i].Id = i;
    processors[i].backgroundLoad = stats->procs[i].bg_walltime;
//    processors[i].backgroundLoad = 0;
    processors[i].computeLoad = 0;
    processors[i].load = processors[i].backgroundLoad;
    processors[i].computeSet = new Set();
    processors[i].pe_speed = stats->procs[i].pe_speed;
//    processors[i].utilization = stats->procs[i].utilization;
    processors[i].available = stats->procs[i].available;
    if (processors[i].available == true) numAvail++;
  }

  int index=0;
  for (i=0; i<stats->n_objs; i++)
  {
  
      LDObjData &odata = stats->objData[i];
      if (odata.migratable == true)
      {
        computes[index].id = odata.objID();
        computes[index].Id = i;
 //       computes[index].handle = odata.handle;
        computes[index].load = odata.wallTime;
        computes[index].processor = -1;
        computes[index].oldProcessor = procs[i];
        computes[index].migratable = odata.migratable;
        if (computes[index].oldProcessor >= P)  {
 	  if (stats->complete_flag) {
            CmiPrintf("LB Panic: the old processor %d of obj %d in RefineKLB cannot be found, is this in a simulation mode?\n", computes[index].oldProcessor, i);
	    CmiAbort("Abort!");
    	  }
          else {
              // an object from outside domain, randomize its location
            computes[i].oldProcessor = CrnRand()%P;
	  }
	}
        index ++;
      }
      else
      {
	// put this compute into background load
	processors[procs[i]].backgroundLoad += odata.wallTime;
	processors[procs[i]].load+= odata.wallTime;
	numComputes--;
      }
  }
  //for (i=0; i < numComputes; i++)
    //  processors[computes[i].oldProcessor].computeLoad += computes[i].load;
}

void RefinerApprox::reinitAssignment()
{
    int i;
    // now numComputes is all the computes: both migratable and nonmigratable.
    // afterwards, nonmigratable computes will be taken off

    for(i=0;i<P;i++)
    {
	Iterator nextCompute;
	nextCompute.id=0;
	computeInfo *c = (computeInfo *)
	    processors[i].computeSet->iterator((Iterator *)&nextCompute);
	while(c)
	{
	    if(c->oldProcessor!=i)
	    {
	    	deAssign(c,&processors[i]);
		if(c->oldProcessor!=-1)
		{
			assign(c,c->oldProcessor);
		}
		else
		{
			assign(c,0);
		}
	    }
	    nextCompute.id++;
	    c = (computeInfo *)
		processors[i].computeSet->next((Iterator *)&nextCompute);
	}
    }
}

void RefinerApprox::multirefine(int num_moves)
{
  computeAverage();
  double avg = averageLoad;
  double max = computeMax();

  int numMoves=0;
  double lbound=avg;
  double ubound=max;

  double EPSILON=0.01;

  numMoves=refine(avg);
  
  if(_lb_debug) CkPrintf("Refined within %d moves\n",numMoves);
  if(numMoves<=num_moves)
      return ;
      
  if(_lb_debug)CkPrintf("[ %lf , %lf ] = %lf > %lf\n",lbound,ubound,ubound-lbound,EPSILON*avg);
  while(ubound-lbound> EPSILON*avg)
  {
      reinitAssignment();
      double testVal=(ubound+lbound)/2;
      numMoves=refine(testVal);
      if(_lb_debug) CkPrintf("Refined within %d moves\n",numMoves);
      if(numMoves>num_moves)
      {
	  lbound=testVal;
      }
      else
      {
	  ubound=testVal;
      }
      if(_lb_debug)CkPrintf("[ %lf , %lf ] = %lf > %lf\n",lbound,ubound,ubound-lbound,EPSILON*avg);
  }
  if(_lb_debug) CkPrintf("Refined within %d moves\n",numMoves);
  return;
}

Set * RefinerApprox::removeBiggestSmallComputes(int num,processorInfo * p,double opt)
{
    int numPComputes=p->computeSet->numElements();
    double totalSmallLoad=0;
    maxHeap *h=new maxHeap(numPComputes);
    Set * removedComputes=new Set();
    int numSmallPComputes=0;
	
    Iterator nextCompute;
    nextCompute.id=0;
    computeInfo *c = (computeInfo *)
	    p->computeSet->iterator((Iterator *)&nextCompute);

    for(int i=0;i<numPComputes;i++)
    {
	if(c->load < opt/2)
	{
	    h->insert((InfoRecord *)c);
	    numSmallPComputes++;
	}
	nextCompute.id++;
	c = (computeInfo *)
	    p->computeSet->next((Iterator *)&nextCompute);
    }

    if(p->backgroundLoad <opt/2)
    {
	totalSmallLoad+=p->backgroundLoad;
    }
    if(numSmallPComputes<num)
    {
	if(_lb_debug)CkPrintf("Error[%d]: Cant remove %d small computes from a total of %d small computes\n",p->Id,num,numSmallPComputes);
    }

    //while(totalSmallLoad > opt/2)
    int j;
    for(j=0;j<num;j++)
    {
	computeInfo *rec=(computeInfo *)(h->deleteMax());
	removedComputes->insert((InfoRecord *)rec);
	totalSmallLoad-=rec->load;
    }

    delete h;
    return removedComputes;
}

Set * RefinerApprox::removeBigComputes(int num,processorInfo * p,double opt)
{
    int numPComputes=p->computeSet->numElements();
    if(num>numPComputes)
    {
	if(_lb_debug)CkPrintf("Error [%d]: Cant remove %d computes out of a total of %d\n",p->Id,num,numPComputes);
	return new Set();
    }
    double totalLoad=p->load;
    maxHeap *h=new maxHeap(numPComputes);
    Set * removedComputes=new Set();
	
    Iterator nextCompute;
    nextCompute.id=0;
    computeInfo *c = (computeInfo *)
	    p->computeSet->iterator((Iterator *)&nextCompute);

    for(int i=0;i<numPComputes;i++)
    {
	h->insert((InfoRecord *)c);
	nextCompute.id++;
	c = (computeInfo *)
	    p->computeSet->next((Iterator *)&nextCompute);
    }

    int j;
    for(j=0;j<num;j++)
    {
	computeInfo *rec=(computeInfo *)(h->deleteMax());
	removedComputes->insert((InfoRecord *)rec);
	totalLoad-=rec->load;
    }

    delete h;
    return removedComputes;
}

double RefinerApprox::getLargestCompute()
{
    double largestC=0;
    for(int i=0;i<P;i++)
    {
	if(processors[i].backgroundLoad > largestC)
	    largestC=processors[i].backgroundLoad;

	Iterator nextCompute;
	nextCompute.id=0;
	computeInfo *c = (computeInfo *)
	    processors[i].computeSet->iterator((Iterator *)&nextCompute);
	while(c)
	{
	    if(c->load > largestC)
	    {
		largestC=c->load;
	    }
	    nextCompute.id++;
	    c = (computeInfo *)
		processors[i].computeSet->next((Iterator *)&nextCompute);
	}
    }
    return largestC;
}

int RefinerApprox::getNumLargeComputes(double opt)
{
    int numLarge=0;
    for(int i=0;i<P;i++)
    {
	if(processors[i].backgroundLoad>=(opt/2))
	    numLarge++;
	Iterator nextCompute;
	nextCompute.id=0;
	computeInfo *c = (computeInfo *)
	    processors[i].computeSet->iterator((Iterator *)&nextCompute);
	int numC=0;
//	CkPrintf("Processor %d \n",i);
	while(c)
	{
	    numC++;
//	    CkPrintf("%d  ",numC);
	    if(c->load>(opt/2))
		numLarge++;
      
	    nextCompute.id++;
	    c = (computeInfo *)
		processors[i].computeSet->next((Iterator *)&nextCompute);
	}
    }
    return numLarge;
    
}

int RefinerApprox::computeA(processorInfo *p,double opt)
{
    int numPComputes=p->computeSet->numElements();
    double totalSmallLoad=0;
    maxHeap *h=new maxHeap(numPComputes);
	
    Iterator nextCompute;
    nextCompute.id=0;
    computeInfo *c = (computeInfo *)
	    p->computeSet->iterator((Iterator *)&nextCompute);

    for(int i=0;i<numPComputes;i++)
    {
	if(c->load < opt/2)
	{
	    totalSmallLoad+=c->load;
	    h->insert((InfoRecord *)c);
	}
	nextCompute.id++;
	c = (computeInfo *)
	    p->computeSet->next((Iterator *)&nextCompute);
    }

    if(p->backgroundLoad <opt/2)
    {
	totalSmallLoad+=p->backgroundLoad;
    }

    int avalue=0;
    while(totalSmallLoad > opt/2)
    {
	avalue++;
	InfoRecord *rec=h->deleteMax();
	totalSmallLoad-=rec->load;
    }
    delete h;
    return avalue;
}

int RefinerApprox::computeB(processorInfo *p,double opt)
{
    int numPComputes=p->computeSet->numElements();
    double totalLoad=p->load;
    if(p->backgroundLoad > opt)
    {
	if(_lb_debug)
	    CkPrintf("Error in computeB: Background load greater than OPT!\n");
	return 0;
    }
    maxHeap *h=new maxHeap(numPComputes);
	
    Iterator nextCompute;
    nextCompute.id=0;
    computeInfo *c = (computeInfo *)
	    p->computeSet->iterator((Iterator *)&nextCompute);

    for(int i=0;i<numPComputes;i++)
    {
	h->insert((InfoRecord*)c);
	nextCompute.id++;
	c = (computeInfo *)
	    p->computeSet->next((Iterator *)&nextCompute);
    }

    int bvalue=0;
    while(totalLoad > opt)
    {
	bvalue++;
	InfoRecord *rec=h->deleteMax();
	totalLoad-=rec->load;
    }

    delete h;
    return bvalue;
}

int RefinerApprox::refine(double opt)
{
    int i;
    if(_lb_debug)CkPrintf("RefinerApprox::refine called with %lf\n",opt);
    if(opt<averageLoad)
	return INFTY;

    int numLargeComputes=getNumLargeComputes(opt);
    if(_lb_debug) CkPrintf("Num of large Computes %d for opt = %10f\n",numLargeComputes,opt);
    if(numLargeComputes>P)
	return INFTY;
    if(getLargestCompute()>opt)
	return INFTY;
    //CkPrintf("Not returning INFTY\n");

    //a[i]= min. number of small jobs to be removed so that total size
    //of remaining small jobs is at most opt/2
    int *a=new int[P];
    //b[i]= min. number of jobs to be removed so that total size
    //of remaining jobs (including large jobs) is at most opt
    int *b=new int[P];
    bool *largeFree=new bool[P];
    Set *largeComputes=new Set();
    Set *smallComputes=new Set();

    //Step 1: Remove all but one large computes on each node
    for(i=0;i<P;i++)
    {
	computeInfo *smallestLargeCompute=NULL;
	largeFree[i]=true;
	Iterator nextCompute;
	nextCompute.id=0;
	computeInfo *c = (computeInfo *)
	    processors[i].computeSet->iterator((Iterator *)&nextCompute);
	while(c)
	{
	    if(c->load>opt/2)
	    {
		largeFree[i]=false;
		largeComputes->insert((InfoRecord*)c);
		deAssign(c,&(processors[i]));
		if (smallestLargeCompute==NULL)
		{
		    smallestLargeCompute=c;
		}
		else if(smallestLargeCompute->load > c->load)
		{
		    smallestLargeCompute=c;
		}
	    }
	    nextCompute.id++;
	    c = (computeInfo *)
		processors[i].computeSet->next((Iterator *)&nextCompute);
	}
	//Check if processor's fixed load is itself large
	if(processors[i].backgroundLoad>opt/2)
	{
	    largeFree[i]=false;
	}
	else 
	{
	    if(smallestLargeCompute)
	    {
		assign(smallestLargeCompute,i);
		largeComputes->remove((InfoRecord*)smallestLargeCompute);
	    }
	}
	if(!largeFree[i]) 
	{
	    if(_lb_debug)
		CkPrintf("Processor %d not LargeFree !\n",i);
	}

	//Step 2: Calculate a[i] and b[i] for each proc.
	a[i]=computeA(&processors[i],opt);
	b[i]=computeB(&processors[i],opt);
    }

    //Step 3: Select L_t(=numLargeComputes) procs with minimum c[i] (=a[i]-b[i]) values.
    //Remove a[i] small jobs from each to get small job load at most opt/2
    minHeap *cHeap=new minHeap(P);
    for(i=0;i<P;i++)
    {
	InfoRecord *ci=new InfoRecord();
	ci->load=a[i]-b[i];
	ci->Id=i;
	cHeap->insert(ci);
    }

    //Set of largeFree procs created with (small jobs < opt/2)
    minHeap *largeFreeLightProcs=new minHeap(P);

    for(i=0;i<numLargeComputes;i++)
    {
	if(_lb_debug) CkPrintf("Removing a large compute %d\n",i);
	//Remove biggest a_k computes from L_t procs
	InfoRecord *cdata= cHeap->deleteMin();
	Set *smallComputesRemoved= removeBiggestSmallComputes(a[cdata->Id],&(processors[cdata->Id]),opt);
	if(largeFree[cdata->Id])
	    largeFreeLightProcs->insert((InfoRecord *)&(processors[cdata->Id]));

	// Keep removed small computes in unassigned set for now
	Iterator nextCompute;
	nextCompute.id=0;
	computeInfo *c=(computeInfo *)
	    smallComputesRemoved->iterator((Iterator*)&nextCompute);
	while(c)
	{
	    deAssign(c,&(processors[cdata->Id]));
	    if(c->load > opt/2)
	    {
		if (_lb_debug) CkPrintf(" Error : Large compute not expected here\n");
	    }
	    else
	    {
		smallComputes->insert((InfoRecord *)c);
	    }
	    nextCompute.id++;
	    c = (computeInfo *)
		smallComputesRemoved->next((Iterator *)&nextCompute);
	}
	delete smallComputesRemoved;
	delete cdata;
    }

    //Step 4 :
    //Remove biggest b computes from P - L_t procs
    // Assign removed large computes to large free procs created in Step 3
    // Keep removed small computes unassigned for now.
    for(i=numLargeComputes;i<P;i++)
    {
	//Remove biggest b computes from P - L_t procs
	InfoRecord *cdata= cHeap->deleteMin();
	Set *computesRemoved=removeBigComputes(b[cdata->Id],&(processors[cdata->Id]),opt);

	// Assign removed large computes to large free procs created in Step 3
	// Keep removed small computes unassigned for now.
	Iterator nextCompute;
	nextCompute.id=0;
	computeInfo *c=(computeInfo *)
	    computesRemoved->iterator((Iterator*)&nextCompute);
	while(c)
	{
	    deAssign(c,&(processors[cdata->Id]));
	    if(c->load > opt/2)
	    {
		processorInfo *targetproc=(processorInfo *)largeFreeLightProcs->deleteMin();
		assign(c,targetproc);
		largeFree[cdata->Id]=true;
		largeFree[targetproc->Id]=false;
	    }
	    else
	    {
		smallComputes->insert((InfoRecord *)c);
	    }
	    nextCompute.id++;
	    c = (computeInfo *)
		computesRemoved->next((Iterator *)&nextCompute);
	}
	delete computesRemoved;
	delete cdata;
    }
    delete cHeap;
    
    //Step 5: Arbitrarily assign remaining largeComputes to large-free procs

    Iterator nextCompute;
    nextCompute.id=0;
    computeInfo *c=(computeInfo *)
      largeComputes->iterator((Iterator*)&nextCompute);
    if(_lb_debug)
    {
	if(c) 
	{
	    CkPrintf("Reassigning Large computes removes in Step 1\n");
	}
	else
	{
	    CkPrintf("No  Large computes removed in Step 1\n");
	}
    }
    while(c)
    {
	/*
	//BUG:: Assign to largeFreeLight Procs instead of largeFree procs
	while(!(largeFree[j]) && j<P-1)
	{
	    j++;
	}
	if(!largeFree[j])
	{
	    if(_lb_debug) CkPrintf("Error finding a large free processor in Step 5\n");
	}
	assign(c,j);
	largeFree[j]=false;
	*/

	processorInfo *targetproc=(processorInfo *)largeFreeLightProcs->deleteMin();
	if(_lb_debug)
	{
	    if(!targetproc)
		CkPrintf("Error finding a large free light proc\n");
	}
	assign(c,targetproc);
	largeFree[targetproc->Id]=false;
	nextCompute.id++;
	c = (computeInfo *)
		largeComputes->next((Iterator *)&nextCompute);
    }

    //Step 6: Assign remaining small jobs one by one to least loaded proc
    minHeap *procsLoad=new minHeap(P);
    for(i=0;i<P;i++)
    {
	procsLoad->insert((InfoRecord *) &(processors[i]) );
    }
    nextCompute.id=0;
    c=(computeInfo *)
	smallComputes->iterator((Iterator*)&nextCompute);
    while(c)
    {
	processorInfo *leastLoadedP=(processorInfo *)procsLoad->deleteMin();
	assign(c,leastLoadedP);
	procsLoad->insert((InfoRecord *)  leastLoadedP);
	nextCompute.id++;
	c = (computeInfo *)
		smallComputes->next((Iterator *)&nextCompute);
    }

    delete largeFreeLightProcs;
    delete procsLoad;
    delete [] a;
    delete [] b;
    delete [] largeFree;
    delete largeComputes;
    delete smallComputes;
    return numMoves();
} 

int RefinerApprox::numMoves()
{
    int nmoves=0;
    for(int i=0;i<numComputes;i++)
    {
	if(computes[i].processor!=computes[i].oldProcessor)
	    nmoves++;
    }
    return nmoves;
}

void RefinerApprox::Refine(int count, CentralLB::LDStats* stats, 
		     int* cur_p, int* new_p, int percentMoves)
{
    
  if(_lb_debug) CkPrintf("\n\n");
  if(_lb_debug) CkPrintf("[%d] RefinerApprox strategy\n",CkMyPe());
  P = count;
  numComputes = stats->n_objs;
  computes = new computeInfo[numComputes];
  processors = new processorInfo[count];

  if(_lb_debug) CkPrintf("Total Number of computes : %d\n",numComputes);

  create(count, stats, cur_p);
  if(_lb_debug) printStats(0);

  int i;
  for (i=0; i<numComputes; i++)
  {
      if(computes[i].oldProcessor!=-1)
      //if(false)
      {
	    assign((computeInfo *) &(computes[i]),
	           (processorInfo *) &(processors[computes[i].oldProcessor]));
      }
      else
      {
	    assign((computeInfo *) &(computes[i]),
	           (processorInfo *) &(processors[0]));
      }
  }
  
  if(_lb_debug) CkPrintf("Total Migratable computes : %d\n\n",numComputes);
  if(_lb_debug) CkPrintf("Total  processors : %d\n",P);
  if(_lb_debug) CkPrintf("Total  available processors : %d\n",numAvail);

  removeComputes();

  computeAverage();

  if(_lb_debug) CkPrintf("Avearge load : %lf\n",averageLoad);
  if(_lb_debug) printStats(0);

 
  int numAllowedMoves=(int)(percentMoves*numComputes/100.0);
  if(numAllowedMoves<0)
    numAllowedMoves=0;
  if(numAllowedMoves>numComputes)
    numAllowedMoves=numComputes;

  if(_lb_args.debug())
  {
    CkPrintf("Percent of allowed moves = %d\n",percentMoves);
    CkPrintf("Number of allowed moves = %d\n",numAllowedMoves);
  }
  //multirefine(numComputes);
  multirefine(numAllowedMoves);

  int nmoves = 0;

  //Initialize new_p[i] to cur_p[i]
  //so that non-migratable computes which
  //are ignored in the calcuation get their
  //new_p asssigned same as cur_p
  for(i=0;i<stats->n_objs;i++)
  {
      new_p[i]=cur_p[i];
  }


  for (int pe=0; pe < P; pe++) 
  {
      Iterator nextCompute;
      nextCompute.id = 0;
      computeInfo *c = (computeInfo *)
	  processors[pe].computeSet->iterator((Iterator *)&nextCompute);
    
      while(c) 
      {
	  new_p[c->Id] = c->processor;
	  if (new_p[c->Id] != cur_p[c->Id]) nmoves++;

	  nextCompute.id++;
	  c = (computeInfo *) processors[pe].computeSet->
	             next((Iterator *)&nextCompute);
      }
  }
  if (_lb_debug) CkPrintf("RefinerApprox: moving %d objects. \n", nmoves);
  delete [] computes;
  delete [] processors;
}


void  RefinerApprox::printStats(int newStats)
{
	
    CkPrintf("%Proc#\tLoad\tObjLoad\tBgdLoad\n");
    for(int i=0;i<P;i++)
    {
	CkPrintf("%d\t\t%lf\t%lf\t%lf\n",i,processors[i].load,processors[i].computeLoad,processors[i].backgroundLoad);
    }
    
}


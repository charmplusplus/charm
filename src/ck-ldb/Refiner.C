// This code is derived from RefineLB.C, and RefineLB.C should
// be rewritten to use this, so there is no code duplication

#include "Refiner.h"

int** Refiner::AllocProcs(int count, CentralLB::LDStats* stats)
{
  int** bufs = new int*[count];

  int total_objs=0;
  int i;
  for(i=0; i<count; i++) 
    total_objs += stats[i].n_objs;

  bufs[0] = new int[total_objs];

  int cur_obj = 0;
  for(i=1; i<count;i++) {
    cur_obj += stats[i-1].n_objs;
    bufs[i] = bufs[0] + cur_obj;
  }
  return bufs;
}

void Refiner::FreeProcs(int** bufs)
{
  delete [] bufs[0];
  delete [] bufs;
}

void Refiner::create(int count, CentralLB::LDStats* stats, int** procs)
{
  int i,j;

  P = count;

  // now numComputes is all the computes: migratable and not.
  // afterwards, nonmigratable computes will be taken off
  numComputes = 0;
  for(j=0; j < P; j++) numComputes+= stats[j].n_objs;
  computes = new computeInfo[numComputes];

  processors = new processorInfo[count];

  int index = 0;
  for(j=0; j < count; j++) {
    processors[j].Id = j;
    processors[j].backgroundLoad = stats[j].bg_cputime;
    processors[j].load = processors[j].backgroundLoad;
    processors[j].computeLoad = 0;
    processors[j].computeSet = new Set();
    processors[j].pe_speed = stats[j].pe_speed;
    processors[j].utilization = stats[j].utilization;
    processors[j].available = stats[j].available;

    LDObjData *odata = stats[j].objData;
    const int osz = stats[j].n_objs;  
    for(i=0; i < osz; i++) {
      if (odata[i].migratable)
      {
        computes[index].id = odata[i].id;
        computes[index].handle = odata[i].handle;
        computes[index].load = odata[i].cpuTime;
        computes[index].originalPE = j;
        computes[index].originalIdx = i;
        computes[index].processor = -1;
        computes[index].oldProcessor = procs[j][i];
        index ++;
      }
      else {
	// if not migratable, add load to processor background
        processors[j].backgroundLoad += odata[i].cpuTime;
	numComputes --;
      }
    }
  }
//  for (i=0; i < numComputes; i++)
//      processors[computes[i].oldProcessor].computeLoad += computes[i].load;
}

void Refiner::assign(computeInfo *c, int processor)
{
  assign(c, &(processors[processor]));
}

void Refiner::assign(computeInfo *c, processorInfo *p)
{
   c->processor = p->Id;
   p->computeSet->insert((InfoRecord *) c);
   p->computeLoad += c->load;
   p->load = p->computeLoad + p->backgroundLoad;
}

void  Refiner::deAssign(computeInfo *c, processorInfo *p)
{
   c->processor = -1;
   p->computeSet->remove(c);
   p->computeLoad -= c->load;
   p->load = p->computeLoad + p->backgroundLoad;
}

void Refiner::computeAverage()
{
  int i;
  double total = 0;
  for (i=0; i<numComputes; i++)
    total += computes[i].load;

  for (i=0; i<P; i++)
    total += processors[i].backgroundLoad;

  averageLoad = total/P;
}

double Refiner::computeMax()
{
  int i;
  double max = processors[0].load;
  for (i=1; i<P; i++) {
    if (processors[i].load > max)
      max = processors[i].load;
  }
  return max;
}

int Refiner::refine()
{
  int finish = 1;
  maxHeap *heavyProcessors = new maxHeap(P);

  Set *lightProcessors = new Set();
  int i;
  for (i=0; i<P; i++) {
    if (processors[i].load > overLoad*averageLoad) {
      //      CkPrintf("Processor %d is HEAVY: load:%f averageLoad:%f!\n",
      //	       i, processors[i].load, averageLoad);
      heavyProcessors->insert((InfoRecord *) &(processors[i]));
    } else if (processors[i].load < averageLoad) {
      //      CkPrintf("Processor %d is LIGHT: load:%f averageLoad:%f!\n",
      //	       i, processors[i].load, averageLoad);
      lightProcessors->insert((InfoRecord *) &(processors[i]));
    }
  }
  int done = 0;

  while (!done) {
    double bestSize;
    computeInfo *bestCompute;
    processorInfo *bestP;
    
    processorInfo *donor = (processorInfo *) heavyProcessors->deleteMax();
    if (!donor) break;

    //find the best pair (c,receiver)
    Iterator nextProcessor;
    processorInfo *p = (processorInfo *) 
      lightProcessors->iterator((Iterator *) &nextProcessor);
    bestSize = 0;
    bestP = 0;
    bestCompute = 0;

    while (p) {
      Iterator nextCompute;
      nextCompute.id = 0;
      computeInfo *c = (computeInfo *) 
	donor->computeSet->iterator((Iterator *)&nextCompute);
      // iout << iINFO << "Considering Procsessor : " 
      //      << p->Id << "\n" << endi;
      while (c) {
	//CkPrintf("c->load: %f p->load:%f overLoad*averageLoad:%f \n",
	//c->load, p->load, overLoad*averageLoad);
	if ( c->load + p->load < overLoad*averageLoad) {
	  // iout << iINFO << "Considering Compute : " 
	  //      << c->Id << " with load " 
	  //      << c->load << "\n" << endi;
	  if(c->load > bestSize) {
	    bestSize = c->load;
	    bestCompute = c;
	    bestP = p;
	  }
	}
	nextCompute.id++;
	c = (computeInfo *) 
	  donor->computeSet->next((Iterator *)&nextCompute);
      }
      p = (processorInfo *) 
	lightProcessors->next((Iterator *) &nextProcessor);
    }

    if (bestCompute) {
      //      CkPrintf("Assign: [%d] with load: %f from %d to %d \n",
      //	       bestCompute->id.id[0], bestCompute->load, 
      //	       donor->Id, bestP->Id);
      deAssign(bestCompute, donor);      
      assign(bestCompute, bestP);
    } else {
      finish = 0;
      break;
    }

    if (bestP->load > averageLoad)
      lightProcessors->remove(bestP);
    
    if (donor->load > overLoad*averageLoad)
      heavyProcessors->insert((InfoRecord *) donor);
    else if (donor->load < averageLoad)
      lightProcessors->insert((InfoRecord *) donor);
  }  
  return finish;
}

void Refiner::Refine(int count, CentralLB::LDStats* stats, 
		     int** cur_p, int** new_p)
{
  //  CkPrintf("[%d] Refiner strategy\n",CkMyPe());

  create(count, stats, cur_p);

  int i;
  for (i=0; i<numComputes; i++)
    assign((computeInfo *) &(computes[i]),
           (processorInfo *) &(processors[computes[i].oldProcessor]));

  computeAverage();

  refine();

  for (int pe=0; pe < P; pe++) {
    Iterator nextCompute;
    nextCompute.id = 0;
    computeInfo *c = (computeInfo *)
      processors[pe].computeSet->iterator((Iterator *)&nextCompute);
    while(c) {
      new_p[c->originalPE][c->originalIdx] = c->processor;
      // if (c->oldProcessor != c->processor)
      //	CkPrintf("Refiner::Refine: from %d to %d\n",
      //		 c->oldProcessor, c->processor);
      nextCompute.id++;
      c = (computeInfo *) processors[pe].computeSet->
	             next((Iterator *)&nextCompute);
    }
  }
};

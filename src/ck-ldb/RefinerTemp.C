/**
 * \addtogroup CkLdb
*/
/*@{*/

/** This code is derived from RefineLB.C, and RefineLB.C should
 be rewritten to use this, so there is no code duplication
*/

#include "RefinerTemp.h"

int* RefinerTemp::AllocProcs(int count, BaseLB::LDStats* stats)
{
  return new int[stats->n_objs];
}

void RefinerTemp::FreeProcs(int* bufs)
{
  delete [] bufs;
}

void RefinerTemp::create(int count, BaseLB::LDStats* stats, int* procs)
{
#ifdef TEMP_LDB
  int i;

  // now numComputes is all the computes: both migratable and nonmigratable.
  // afterwards, nonmigratable computes will be taken off

  numAvail = 0;
  for(i=0; i < P; i++) {
    processors[i].Id = i;
    processors[i].backgroundLoad = stats->procs[i].bg_walltime;
    processors[i].load = processors[i].backgroundLoad;
    processors[i].computeLoad = 0;
    processors[i].computeSet = new Set();
    processors[i].pe_speed = stats->procs[i].pe_speed;
//    processors[i].utilization = stats->procs[i].utilization;
    processors[i].available = stats->procs[i].available;
    if (processors[i].available == true) numAvail++;
  }

  for (i=0; i<stats->n_objs; i++)
  {
	LDObjData &odata = stats->objData[i];
	computes[i].Id = i;
        computes[i].id = odata.objID();
//        computes[i].handle = odata.handle;
        computes[i].load = odata.wallTime;     // was cpuTime
        computes[i].processor = -1;
        computes[i].oldProcessor = procs[i];
				computes[i].omid = odata.omID().id.idx;
        computes[i].migratable = odata.migratable;
        if (computes[i].oldProcessor >= P)  {
 	  if (stats->complete_flag)
            CmiAbort("LB Panic: the old processor in RefineLB cannot be found, is this in a simulation mode?");
          else {
              // an object from outside domain, randomize its location
            computes[i].oldProcessor = CrnRand()%P;
	  }
	}
  }
//  for (i=0; i < numComputes; i++)
//      processors[computes[i].oldProcessor].computeLoad += computes[i].load;
#endif
}

void RefinerTemp::assign(computeInfo *c, int processor)
{
  assign(c, &(processors[processor]));
}

void RefinerTemp::assign(computeInfo *c, processorInfo *p)
{
   c->processor = p->Id;
   p->computeSet->insert((InfoRecord *) c);
  int oldPe=c->oldProcessor;
   p->computeLoad += c->load*procFreq[oldPe];
   p->load = p->computeLoad + p->backgroundLoad*procFreq[p->Id];
/*
          int ind1 = c->id.getID()[0];
        int ind2 = c->id.getID()[1];
	if(ind1==0 && ind2==48) CkPrintf("----- assigning to proc%d load:%f\n",p->Id,p->computeLoad);
*/
}

void  RefinerTemp::deAssign(computeInfo *c, processorInfo *p)
{
   c->processor = -1;
   p->computeSet->remove(c);
int oldPe=c->oldProcessor;
   p->computeLoad -= c->load*procFreq[p->Id];
   p->load = p->computeLoad + p->backgroundLoad*procFreq[p->Id];
}

void RefinerTemp::computeAverage()
{
  int i;
  double total = 0.;
  for (i=0; i<numComputes; i++) total += computes[i].load*procFreq[computes[i].oldProcessor];

  for (i=0; i<P; i++)
    if (processors[i].available == true)
        total += processors[i].backgroundLoad*procFreq[processors[i].Id];

  averageLoad = total/numAvail;
totalInst=total;
}

double RefinerTemp::computeMax()
{
  int i;
  double max = -1.0;
  for (i=0; i<P; i++) {
    if (processors[i].available == true && processors[i].load > max)
//      max = processors[i].load;
	max=processors[i].load/procFreqNew[processors[i].Id];
  }
  return max;
}

double RefinerTemp::computeMax(int *maxPe)
{
  int i;
  double max = -1.0,maxratio=-1.0;
  for (i=0; i<P; i++) {
//CkPrintf(" ********** pe%d load=%f freq=%d ratio=%f\n",processors[i].Id,processors[i].load,procFreqNew[processors[i].Id],processors[i].load/procFreqNew[processors[i].Id]);
    if (processors[i].available == true && processors[i].load/procFreqNew[processors[i].Id] > maxratio)
    {
//      max = processors[i].load;
//CkPrintf(" ********** pe%d load=%f freq=%d \n",processors[i].Id,processors[i].load,procFreqNew[processors[i].Id]);
        maxratio=processors[i].load/procFreqNew[processors[i].Id];
        max=processors[i].load;
        *maxPe=processors[i].Id;
    }
  }
  return max;
}

int RefinerTemp::isHeavy(processorInfo *p)
{
  if (p->available == true) 
//     return p->load > overLoad*averageLoad;
	return p->load > overLoad*(totalInst*procFreqNew[p->Id]/sumFreqs);
  else {
     return p->computeSet->numElements() != 0;
  }
}

int RefinerTemp::isLight(processorInfo *p)
{
  if (p->available == true) 
//     return p->load < averageLoad;
	return p->load < totalInst*procFreqNew[p->Id]/sumFreqs;
  else 
     return 0;
}

// move the compute jobs out from unavailable PE
void RefinerTemp::removeComputes()
{
  int first;
  Iterator nextCompute;

  if (numAvail < P) {
    if (numAvail == 0) CmiAbort("No processor available!");
    for (first=0; first<P; first++)
      if (processors[first].available == true) break;
    for (int i=0; i<P; i++) {
      if (processors[i].available == false) {
          computeInfo *c = (computeInfo *)
	           processors[i].computeSet->iterator((Iterator *)&nextCompute);
	  while (c) {
	    deAssign(c, &processors[i]);
	    assign(c, &processors[first]);
	    nextCompute.id++;
            c = (computeInfo *)
	           processors[i].computeSet->next((Iterator *)&nextCompute);
	  }
      }
    }
  }
}

int RefinerTemp::refine()
{
#ifdef TEMP_LDB
  int i;
  int finish = 1;
  maxHeap *heavyProcessors = new maxHeap(P);

  Set *lightProcessors = new Set();
  for (i=0; i<P; i++) {
    if (isHeavy(&processors[i])) {  
//            CPrintf("Processor %d is HEAVY: load:%f averageLoad:%f!\n",
//      	       i, processors[i].load, averageLoad);
      heavyProcessors->insert((InfoRecord *) &(processors[i]));
    } else if (isLight(&processors[i])) {
//            CkPrintf("Processor %d is LIGHT: load:%f averageLoad:%f!\n",
//      	       i, processors[i].load, averageLoad);
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
          int ind1 = c->id.getID()[0];
        int ind2 = c->id.getID()[1];

        if (!c->migratable) {
	  nextCompute.id++;
	  c = (computeInfo *) 
	    donor->computeSet->next((Iterator *)&nextCompute);
          continue;
        }
//				else CkPrintf("c->id:%f\n",c->load);
//	CkPrintf("c->load: %f p->load:%f overLoad*averageLoad:%f \n",
//	c->load, p->load, overLoad*averageLoad);
//	if ( c->load + p->load < overLoad*averageLoad) {

	if ( c->load*procFreq[c->oldProcessor] + p->load < overLoad*(totalInst*procFreqNew[p->Id]/sumFreqs)
       && ind1!=0 && ind2!=0/* && ind2>=10*/) {
	  // iout << iINFO << "Considering Compute : " 
	  //      << c->Id << " with load " 
	  //      << c->load << "\n" << endi;
//	  if(c->load > bestSize) {
	if(c->load*procFreq[c->oldProcessor] > bestSize/* &&  (c->omid==10 && procFreq[donor->Id]>procFreqNew[donor->Id])*/) {
//	CkPrintf("c:%d is going to PE%d load:%d\n",c->Id,p->Id,c->load);
	bestSize = c->load*procFreq[c->oldProcessor];
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
//	CkPrintf("best load:%f\n",bestCompute->load);

    if (bestCompute) {
//CkPrintf("best load:%f\n",bestCompute->load);
//CkPrintf("TTT c->omid:%d c->load:%f P#%d -> P#%d %d -> %d\n",bestCompute->omid,bestCompute->load,donor->Id,bestP->Id,procFreq[donor->Id],procFreqNew[donor->Id]);
      //      CkPrintf("Assign: [%d] with load: %f from %d to %d \n",
      //	       bestCompute->id.id[0], bestCompute->load, 
      //	       donor->Id, bestP->Id);
      deAssign(bestCompute, donor);      
      assign(bestCompute, bestP);
    } else {
      finish = 0;
      break;
    }

//    if (bestP->load > averageLoad)
	if(bestP->load > totalInst*procFreqNew[bestP->Id]/sumFreqs)
      lightProcessors->remove(bestP);
    
    if (isHeavy(donor))
      heavyProcessors->insert((InfoRecord *) donor);
    else if (isLight(donor))
      lightProcessors->insert((InfoRecord *) donor);
  }  

  delete heavyProcessors;
  delete lightProcessors;

  return finish;
#else
	return 0;
#endif
}

int RefinerTemp::multirefine()
{
  computeAverage();
  double avg = averageLoad;
  int maxPe=-1;
 // double max = computeMax();
  double max = computeMax(&maxPe);

  //const double overloadStep = 0.01;
  const double overloadStep = 0.01;
  const double overloadStart = 1.001;
//  double dCurOverload = max / avg;
  double dCurOverload = max /(totalInst*procFreqNew[maxPe]/sumFreqs); 
                                                                               
  int minOverload = 0;
  int maxOverload = (int)((dCurOverload - overloadStart)/overloadStep + 1);
  double dMinOverload = minOverload * overloadStep + overloadStart;
  double dMaxOverload = maxOverload * overloadStep + overloadStart;
  int curOverload;
  int refineDone = 0;
//CmiPrintf("maxPe=%d max=%f myAvg=%f dMinOverload: %f dMaxOverload: %f\n",maxPe,max,(totalInst*procFreqNew[maxPe]/sumFreqs), dMinOverload, dMaxOverload);

  if (_lb_args.debug()>=1)
    CmiPrintf("dMinOverload: %f dMaxOverload: %f\n", dMinOverload, dMaxOverload);
                                                                                
  overLoad = dMinOverload;
  if (refine())
    refineDone = 1;
  else {
    overLoad = dMaxOverload;
    if (!refine()) {
      CmiPrintf("ERROR: Could not refine at max overload\n");
      refineDone = 1;
    }
  }
                                                                                
  // Scan up, until we find a refine that works
  while (!refineDone) {
    if (maxOverload - minOverload <= 1)
      refineDone = 1;
    else {
      curOverload = (maxOverload + minOverload ) / 2;
                                                                                
      overLoad = curOverload * overloadStep + overloadStart;
      if (_lb_args.debug()>=1)
      CmiPrintf("Testing curOverload %d = %f [min,max]= %d, %d\n", curOverload, overLoad, minOverload, maxOverload);
      if (refine())
        maxOverload = curOverload;
      else
        minOverload = curOverload;
    }
  }
  return 1;
}


  RefinerTemp::RefinerTemp(
double _overload,int *p,int *pn,int nProcs) {
P=nProcs;
overLoad = _overload; computes=0; processors=0;
procFreq=p;procFreqNew=pn;
sumFreqs=0;
for(int i=0;i<P;i++)
{
sumFreqs+=procFreqNew[i];
}

  }

void RefinerTemp::Refine(int count, BaseLB::LDStats* stats, 
		     int* cur_p, int* new_p)
{
#ifdef TEMP_LDB
  //  CkPrintf("[%d] RefinerTemp strategy\n",CkMyPe());

  P = count;
  numComputes = stats->n_objs;
  computes = new computeInfo[numComputes];
  processors = new processorInfo[count];

  create(count, stats, cur_p);

  int i;
  for (i=0; i<numComputes; i++)
	{
          int ind1 = computes[i].id.getID()[0];
        int ind2 = computes[i].id.getID()[1];
  if(ind1==0 && ind2==48) CkPrintf("----- assigning oldproc%d load:%f\n",computes[i].oldProcessor,computes[i].load);

    assign((computeInfo *) &(computes[i]),
           (processorInfo *) &(processors[computes[i].oldProcessor]));
	}
  removeComputes();

  computeAverage();

  if (_lb_args.debug()>2)  {
    CkPrintf("Old PE load (bg load): ");
    for (i=0; i<count; i++) CkPrintf("%d:%f(%f) ", i, processors[i].load, processors[i].backgroundLoad);
    CkPrintf("\n");
  }

  multirefine();

  int nmoves = 0;
  for (int pe=0; pe < P; pe++) {
    Iterator nextCompute;
    nextCompute.id = 0;
    computeInfo *c = (computeInfo *)
      processors[pe].computeSet->iterator((Iterator *)&nextCompute);
    while(c) {
      new_p[c->Id] = c->processor;
      if (new_p[c->Id] != cur_p[c->Id]) nmoves++;
//      if (c->oldProcessor != c->processor)
//      CkPrintf("RefinerTemp::Refine: from %d to %d\n", c->oldProcessor, c->processor);
      nextCompute.id++;
      c = (computeInfo *) processors[pe].computeSet->
	             next((Iterator *)&nextCompute);
    }
  }
  if (_lb_args.debug()>2)  {
    CkPrintf("New PE load: ");
    for (i=0; i<count; i++) CkPrintf("%f ", processors[i].load);
    CkPrintf("\n");
  }
  if (_lb_args.debug()>1) 
    CkPrintf("RefinerTemp: moving %d obejcts. \n", nmoves);
  delete [] computes;
  delete [] processors;
#endif
}


/*@}*/

/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

/** This code is derived from RefineLB.C, and RefineLB.C should
 be rewritten to use this, so there is no code duplication
*/

#include "RefinerComm.h"


void RefinerComm::create(int count, CentralLB::LDStats* stats, int* procs)
{
  Refiner::create(count, stats, procs);

  // fill communication
  stats->makeCommHash();
  LDCommData *cdata = stats->commData;
  for (int i=0; i<stats->n_comm; i++) 
  {
     	LDCommData &comm = cdata[i];
	if (comm.from_proc()) continue;
	int computeIdx = stats->getHash(comm.sender);
        CmiAssert(computeIdx >= 0 && computeIdx < numComputes);
        computes[computeIdx].messages.push_back(i);
  }
}

void RefinerComm::computeAverage()
{
  int i;
  double total = 0;
  for (i=0; i<numComputes; i++) total += computes[i].load;

  for (i=0; i<P; i++)
    if (processors[i].available == CmiTrue) 
	total += processors[i].backgroundLoad;

  averageLoad = total/numAvail;
}

void RefinerComm::commSummary(CentralLB::LDStats* stats)
{
  int i;
  msgSentCount = new int[P]; // # of messages sent by each PE
  msgRecvCount = new int[P]; // # of messages received by each PE
  byteSentCount = new int[P];// # of bytes sent by each PE
  byteRecvCount = new int[P];// # of bytes reeived by each PE

  for(i = 0; i < P; i++)
    msgSentCount[i] = msgRecvCount[i] = byteSentCount[i] = byteRecvCount[i] = 0;

  for (int cidx=0; cidx < stats->n_comm; cidx++) {
    LDCommData& cdata = stats->commData[cidx];
    int senderPE, receiverPE;
    if (cdata.from_proc())
      senderPE = cdata.src_proc;
    else {
      int idx = stats->getHash(cdata.sender);
      CmiAssert(idx != -1);
      senderPE = stats->from_proc[idx];	       // object's original processor
      CmiAssert(senderPE != -1);
    }
    if (cdata.receiver.get_type() == LD_PROC_MSG)
      receiverPE = cdata.receiver.proc();
    else {
      int idx = stats->getHash(cdata.receiver.get_destObj());
      CmiAssert(idx != -1);
      receiverPE = stats->from_proc[idx];
      CmiAssert(receiverPE != -1);
    }
    if(senderPE != receiverPE)
    {
       msgSentCount[senderPE] += cdata.messages;
       byteSentCount[senderPE] += cdata.bytes;

       msgRecvCount[receiverPE] += cdata.messages;
       byteRecvCount[receiverPE] += cdata.bytes;
     }
  }
}

double RefinerComm::commCost(int pe)
{
  return msgRecvCount[pe]  * PER_MESSAGE_RECV_OVERHEAD +
	 msgSentCount[pe]  * PER_MESSAGE_SEND_OVERHEAD +
	 byteRecvCount[pe] * PER_BYTE_RECV_OVERHEAD +
	 byteSentCount[pe] * PER_BYTE_SEND_OVERHEAD;
}

int RefinerComm::refine()
{
  int i;
  int finish = 1;
  maxHeap *heavyProcessors = new maxHeap(P);

  Set *lightProcessors = new Set();
  for (i=0; i<P; i++) {
    if (isHeavy(&processors[i])) {  
      //      CkPrintf("Processor %d is HEAVY: load:%f averageLoad:%f!\n",
      //	       i, processors[i].load, averageLoad);
      heavyProcessors->insert((InfoRecord *) &(processors[i]));
    } else if (isLight(&processors[i])) {
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
        if (!c->migratable) {
	  nextCompute.id++;
	  c = (computeInfo *) 
	    donor->computeSet->next((Iterator *)&nextCompute);
          continue;
        }
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
    
    if (isHeavy(donor))
      heavyProcessors->insert((InfoRecord *) donor);
    else if (isLight(donor))
      lightProcessors->insert((InfoRecord *) donor);
  }  

  delete heavyProcessors;
  delete lightProcessors;

  return finish;
}

void RefinerComm::Refine(int count, CentralLB::LDStats* stats, 
		     int* cur_p, int* new_p)
{
  //  CkPrintf("[%d] Refiner strategy\n",CkMyPe());

  P = count;
  numComputes = stats->n_objs;
  computes = new computeInfo[numComputes];
  processors = new processorInfo[count];

  create(count, stats, cur_p);

  int i;
  for (i=0; i<numComputes; i++)
    assign((computeInfo *) &(computes[i]),
           (processorInfo *) &(processors[computes[i].oldProcessor]));

  removeComputes();

  computeAverage();

  refine();

  for (int pe=0; pe < P; pe++) {
    Iterator nextCompute;
    nextCompute.id = 0;
    computeInfo *c = (computeInfo *)
      processors[pe].computeSet->iterator((Iterator *)&nextCompute);
    while(c) {
      new_p[c->originalIdx] = c->processor;
//      if (c->oldProcessor != c->processor)
//      CkPrintf("Refiner::Refine: from %d to %d\n", c->oldProcessor, c->processor);
      nextCompute.id++;
      c = (computeInfo *) processors[pe].computeSet->
	             next((Iterator *)&nextCompute);
    }
  }
  delete [] msgSentCount;

  delete [] computes;
  delete [] processors;
};


/*@}*/

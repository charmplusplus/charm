/**
 * \addtogroup CkLdb
*/
/*@{*/

/** This code is derived from RefineLB.C, and RefineLB.C should
 be rewritten to use this, so there is no code duplication
*/

#include "elements.h"
#include "ckheap.h"
#include "RefinerComm.h"


void RefinerComm::create(int count, BaseLB::LDStats* _stats, int* procs)
{
  int i;
  stats = _stats;
  Refiner::create(count, _stats, procs);

  for (i=0; i<stats->n_comm; i++) 
  {
     	LDCommData &comm = stats->commData[i];
	if (!comm.from_proc()) {
          // out going message
	  int computeIdx = stats->getSendHash(comm);
          CmiAssert(computeIdx >= 0 && computeIdx < numComputes);
          computes[computeIdx].sendmessages.push_back(i);
        }

	// FIXME: only obj msg here
        // incoming messages
	if (comm.receiver.get_type() == LD_OBJ_MSG)  {
          int computeIdx = stats->getRecvHash(comm);
          CmiAssert(computeIdx >= 0 && computeIdx < numComputes);
          computes[computeIdx].recvmessages.push_back(i);
        }
  }
}

void RefinerComm::computeAverage()
{
  int i;
  double total = 0.;
  for (i=0; i<numComputes; i++) total += computes[i].load;

  for (i=0; i<P; i++) {
    if (processors[i].available == true) {
	total += processors[i].backgroundLoad;
        total += commTable->overheadOnPe(i);
    }
  }

  averageLoad = total/numAvail;
}

// compute the initial per processor communication overhead
void RefinerComm::processorCommCost()
{
  int i;

  for (int cidx=0; cidx < stats->n_comm; cidx++) {
    LDCommData& cdata = stats->commData[cidx];
    int senderPE = -1, receiverPE = -1;
    if (cdata.from_proc())
      senderPE = cdata.src_proc;
    else {
      int idx = stats->getSendHash(cdata);
      CmiAssert(idx != -1);
      senderPE = computes[idx].oldProcessor;	// object's original processor
    }
    CmiAssert(senderPE != -1);
    int ctype = cdata.receiver.get_type();
    if (ctype==LD_PROC_MSG || ctype==LD_OBJ_MSG) {
      if (ctype==LD_PROC_MSG)
        receiverPE = cdata.receiver.proc();
      else {    // LD_OBJ_MSG
        int idx = stats->getRecvHash(cdata);
        CmiAssert(idx != -1);
        receiverPE = computes[idx].oldProcessor;
      }
      CmiAssert(receiverPE != -1);
      if(senderPE != receiverPE)
      {
        commTable->increase(true, senderPE, cdata.messages, cdata.bytes);
        commTable->increase(false, receiverPE, cdata.messages, cdata.bytes);
      }
    }
    else if (ctype == LD_OBJLIST_MSG) {
      int nobjs;
      LDObjKey *objs = cdata.receiver.get_destObjs(nobjs);
      for (i=0; i<nobjs; i++) {
        int idx = stats->getHash(objs[i]);
        if(idx == -1)
             if (_lb_args.migObjOnly()) continue;
             else CkAbort("Error in search\n");
        receiverPE = computes[idx].oldProcessor;
        CmiAssert(receiverPE != -1);
        if(senderPE != receiverPE)
        {
          commTable->increase(true, senderPE, cdata.messages, cdata.bytes);
          commTable->increase(false, receiverPE, cdata.messages, cdata.bytes);
        }
      }
    }
  }
  // recalcualte the cpu load
  for (i=0; i<P; i++) 
  {
    processorInfo *p = &processors[i];
    p->load = p->computeLoad + p->backgroundLoad + commTable->overheadOnPe(i);
  }
}

void RefinerComm::assign(computeInfo *c, int processor)
{
  assign(c, &(processors[processor]));
}

void RefinerComm::assign(computeInfo *c, processorInfo *p)
{
   c->processor = p->Id;
   p->computeSet->insert((InfoRecord *) c);
   p->computeLoad += c->load;
//   p->load = p->computeLoad + p->backgroundLoad;
   // add communication cost
   Messages m;
   objCommCost(c->Id, p->Id, m);
   commTable->increase(true, p->Id, m.msgSent, m.byteSent);
   commTable->increase(false, p->Id, m.msgRecv, m.byteRecv);

//   CmiPrintf("Assign %d to %d commCost: %d %d %d %d \n", c->Id, p->Id, byteSent,msgSent,byteRecv,msgRecv);

   commAffinity(c->Id, p->Id, m);
   commTable->increase(false, p->Id, -m.msgSent, -m.byteSent);
   commTable->increase(true, p->Id, -m.msgRecv, -m.byteRecv);   // reverse

//   CmiPrintf("Assign %d to %d commAffinity: %d %d %d %d \n", c->Id, p->Id, -byteSent,-msgSent,-byteRecv,-msgRecv);

   p->load = p->computeLoad + p->backgroundLoad + commTable->overheadOnPe(p->Id);
}

void  RefinerComm::deAssign(computeInfo *c, processorInfo *p)
{
//   c->processor = -1;
   p->computeSet->remove(c);
   p->computeLoad -= c->load;
//   p->load = p->computeLoad + p->backgroundLoad;
   Messages m;
   objCommCost(c->Id, p->Id, m);
   commTable->increase(true, p->Id, -m.msgSent, -m.byteSent);
   commTable->increase(false, p->Id, -m.msgRecv, -m.byteRecv);
   
   commAffinity(c->Id, p->Id, m);
   commTable->increase(true, p->Id, m.msgSent, m.byteSent);
   commTable->increase(false, p->Id, m.msgRecv, m.byteRecv);

   p->load = p->computeLoad + p->backgroundLoad + commTable->overheadOnPe(p->Id);
}

// how much communication from compute c  to pe
// byteSent, msgSent are messages from object c to pe p
// byteRecv, msgRecv are messages from pe p to obejct c
void RefinerComm::commAffinity(int c, int pe, Messages &m)
{
  int i;
  m.clear();
  computeInfo &obj = computes[c];

  int nSendMsgs = obj.sendmessages.length();
  for (i=0; i<nSendMsgs; i++) {
    LDCommData &cdata = stats->commData[obj.sendmessages[i]];
    bool sendtope = false;
    if (cdata.receiver.get_type() == LD_OBJ_MSG) {
      int recvCompute = stats->getRecvHash(cdata);
      int recvProc = computes[recvCompute].processor;
      if (recvProc != -1 && recvProc == pe) sendtope = true;
    }
    else if (cdata.receiver.get_type() == LD_OBJLIST_MSG) {  // multicast
      int nobjs;
      LDObjKey *recvs = cdata.receiver.get_destObjs(nobjs);
      for (int j=0; j<nobjs; j++) {
        int recvCompute = stats->getHash(recvs[j]);
        int recvProc = computes[recvCompute].processor;	// FIXME
        if (recvProc != -1 && recvProc == pe) { sendtope = true; continue; }
      }  
    }
    if (sendtope) {
      m.byteSent += cdata.bytes;
      m.msgSent += cdata.messages;
    }
  }  // end of for

  int nRecvMsgs = obj.recvmessages.length();
  for (i=0; i<nRecvMsgs; i++) {
    LDCommData &cdata = stats->commData[obj.recvmessages[i]];
    int sendProc;
    if (cdata.from_proc()) {
      sendProc = cdata.src_proc;
    }
    else {
      int sendCompute = stats->getSendHash(cdata);
      sendProc = computes[sendCompute].processor;
    }
    if (sendProc != -1 && sendProc == pe) {
      m.byteRecv += cdata.bytes;
      m.msgRecv += cdata.messages;
    }
  }  // end of for
}

// assume c is on pe, how much comm overhead it will be?
void RefinerComm::objCommCost(int c, int pe, Messages &m)
{
  int i;
  m.clear();
  computeInfo &obj = computes[c];

  // find out send overhead for every outgoing message that has receiver
  // not same as pe
  int nSendMsgs = obj.sendmessages.length();
  for (i=0; i<nSendMsgs; i++) {
    LDCommData &cdata = stats->commData[obj.sendmessages[i]];
    bool diffPe = false;
    if (cdata.receiver.get_type() == LD_PROC_MSG) {
      CmiAssert(0);
    }
    if (cdata.receiver.get_type() == LD_OBJ_MSG) {
      int recvCompute = stats->getRecvHash(cdata);
      int recvProc = computes[recvCompute].processor;
      if (recvProc!= -1 && recvProc != pe) diffPe = true;
    }
    else if (cdata.receiver.get_type() == LD_OBJLIST_MSG) {  // multicast
      int nobjs;
      LDObjKey *recvs = cdata.receiver.get_destObjs(nobjs);
      for (int j=0; j<nobjs; j++) {
        int recvCompute = stats->getHash(recvs[j]);
        int recvProc = computes[recvCompute].processor;	// FIXME
        if (recvProc!= -1 && recvProc != pe) { diffPe = true; }
      }  
    }
    if (diffPe) {
      m.byteSent += cdata.bytes;
      m.msgSent += cdata.messages;
    }
  }  // end of for

  // find out recv overhead for every incoming message that has sender
  // not same as pe
  int nRecvMsgs = obj.recvmessages.length();
  for (i=0; i<nRecvMsgs; i++) {
    LDCommData &cdata = stats->commData[obj.recvmessages[i]];
    bool diffPe = false;
    if (cdata.from_proc()) {
      if (cdata.src_proc != pe) diffPe = true;
    }
    else {
      int sendCompute = stats->getSendHash(cdata);
      int sendProc = computes[sendCompute].processor;
      if (sendProc != -1 && sendProc != pe) diffPe = true;
    }
    if (diffPe) {	// sender is not pe
      m.byteRecv += cdata.bytes;
      m.msgRecv += cdata.messages;
    }
  }  // end of for
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
     // 	       i, processors[i].load, averageLoad);
      heavyProcessors->insert((InfoRecord *) &(processors[i]));
    } else if (isLight(&processors[i])) {
      //      CkPrintf("Processor %d is LIGHT: load:%f averageLoad:%f!\n",
     // 	       i, processors[i].load, averageLoad);
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
    bestP = NULL;
    bestCompute = NULL;

    while (p) {
      Iterator nextCompute;
      nextCompute.id = 0;
      computeInfo *c = (computeInfo *) 
	donor->computeSet->iterator((Iterator *)&nextCompute);
      //CmiPrintf("Considering Procsessor : %d with load: %f for donor: %d\n", p->Id, p->load, donor->Id);
      while (c) {
        if (!c->migratable) {
	  nextCompute.id++;
	  c = (computeInfo *) 
	    donor->computeSet->next((Iterator *)&nextCompute);
          continue;
        }
	//CkPrintf("c->load: %f p->load:%f overLoad*averageLoad:%f \n",
	//c->load, p->load, overLoad*averageLoad);
        Messages m;
        objCommCost(c->Id, donor->Id, m);
        double commcost = m.cost();
        commAffinity(c->Id, p->Id, m);
        double commgain = m.cost();;

        //CmiPrintf("Considering Compute: %d with load %f commcost:%f commgain:%f\n", c->Id, c->load, commcost, commgain);
	if ( c->load + p->load + commcost - commgain < overLoad*averageLoad) {
          //CmiPrintf("[%d] comm gain %f bestSize:%f\n", c->Id, commgain, bestSize);
	  if(c->load + commcost - commgain > bestSize) {
	    bestSize = c->load + commcost - commgain;
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
      if (_lb_args.debug())
        CkPrintf("Assign: [%d] with load: %f from %d to %d \n",
      	       bestCompute->Id, bestCompute->load, 
               donor->Id, bestP->Id);
      deAssign(bestCompute, donor);      
      assign(bestCompute, bestP);

      // show the load
      if (_lb_args.debug())  printLoad();

      // update commnication
      computeAverage();
      delete heavyProcessors;
      delete lightProcessors;
      heavyProcessors = new maxHeap(P);
      lightProcessors = new Set();
      for (i=0; i<P; i++) {
        if (isHeavy(&processors[i])) {  
          //      CkPrintf("Processor %d is HEAVY: load:%f averageLoad:%f!\n",
          //	       i, processors[i].load, averageLoad);
          heavyProcessors->insert((InfoRecord *) &(processors[i]));
        } else if (isLight(&processors[i])) {
          lightProcessors->insert((InfoRecord *) &(processors[i]));
        }
      }
      if (_lb_args.debug()) CmiPrintf("averageLoad after assignment: %f\n", averageLoad);
    } else {
      finish = 0;
      break;
    }


/*
    if (bestP->load > averageLoad)
      lightProcessors->remove(bestP);
    
    if (isHeavy(donor))
      heavyProcessors->insert((InfoRecord *) donor);
    else if (isLight(donor))
      lightProcessors->insert((InfoRecord *) donor);
*/
  }  

  delete heavyProcessors;
  delete lightProcessors;

  return finish;
}

void RefinerComm::Refine(int count, BaseLB::LDStats* stats, 
		     int* cur_p, int* new_p)
{
  //  CkPrintf("[%d] Refiner strategy\n",CkMyPe());

  P = count;
  numComputes = stats->n_objs;
  computes = new computeInfo[numComputes];
  processors = new processorInfo[count];
  commTable = new CommTable(P);

  // fill communication hash table
  stats->makeCommHash();

  create(count, stats, cur_p);

  int i;
  for (i=0; i<numComputes; i++)
    assign((computeInfo *) &(computes[i]),
           (processorInfo *) &(processors[computes[i].oldProcessor]));

  commTable->clear();

  // recalcualte the cpu load
  processorCommCost();

  removeComputes();
  if (_lb_args.debug())  printLoad();

  computeAverage();
  if (_lb_args.debug()) CmiPrintf("averageLoad: %f\n", averageLoad);

  multirefine();

  for (int pe=0; pe < P; pe++) {
    Iterator nextCompute;
    nextCompute.id = 0;
    computeInfo *c = (computeInfo *)
      processors[pe].computeSet->iterator((Iterator *)&nextCompute);
    while(c) {
      new_p[c->Id] = c->processor;
//      if (c->oldProcessor != c->processor)
//      CkPrintf("Refiner::Refine: from %d to %d\n", c->oldProcessor, c->processor);
      nextCompute.id++;
      c = (computeInfo *) processors[pe].computeSet->
	             next((Iterator *)&nextCompute);
    }
  }

  delete [] computes;
  delete [] processors;
  delete commTable;
}

RefinerComm::CommTable::CommTable(int P)
{
  count = P;
  msgSentCount = new int[P]; // # of messages sent by each PE
  msgRecvCount = new int[P]; // # of messages received by each PE
  byteSentCount = new int[P];// # of bytes sent by each PE
  byteRecvCount = new int[P];// # of bytes reeived by each PE
  clear();
}

RefinerComm::CommTable::~CommTable()
{
  delete [] msgSentCount;
  delete [] msgRecvCount;
  delete [] byteSentCount;
  delete [] byteRecvCount;
}

void RefinerComm::CommTable::clear()
{
  for(int i = 0; i < count; i++)
    msgSentCount[i] = msgRecvCount[i] = byteSentCount[i] = byteRecvCount[i] = 0;
}

void RefinerComm::CommTable::increase(bool issend, int pe, int msgs, int bytes)
{
  if (issend) {
    msgSentCount[pe] += msgs;
    byteSentCount[pe] += bytes;
  }
  else {
    msgRecvCount[pe] += msgs;
    byteRecvCount[pe] += bytes;
  }
}

double RefinerComm::CommTable::overheadOnPe(int pe)
{
  return msgRecvCount[pe]  * PER_MESSAGE_RECV_OVERHEAD +
	 msgSentCount[pe]  * _lb_args.alpha() +
	 byteRecvCount[pe] * PER_BYTE_RECV_OVERHEAD +
	 byteSentCount[pe] * _lb_args.beta();
}

/*@}*/

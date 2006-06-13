#include <unistd.h>
#include "onesided.h"

CProxy_Main mainProxy;

Main::Main(CkArgMsg* m)
{
  int numProcs = 2;
  CProxy_commtest arr = CProxy_commtest::ckNew(2);
  mainProxy = thisProxy;

  if(m->argc!=3) {
    CkPrintf("USAGE: pgm <typeOfRmaOp> <with/wo Callback>\n");
    CkExit();
  }
  arr[0].startRMA(atoi(m->argv[1]), atoi(m->argv[2]));
  delete m;
}

void Main::done(void)
{
  CkExit();
}


commtest::commtest()
{
  idx = CkMyPe();
  dest = (idx==0)?1:0;
  srcAddr = NULL;
  destAddr = NULL;
  size = 50;
  srcChar = (idx==0)?'1':'2';
  destChar = (idx==0)?'2':'1';
  operation = 0;
  callb = 0;
  CkPrintf("[%d]Object created on %d\n",idx,idx);
  CpvAccess(_cmvar) = this;
}

void commtest::startRMA(int op, int cb) {
  operation = op;
  callb = cb;
  srcAddr = (char*)malloc(size*sizeof(char));
  //srcAddr = (char*)CmiDMAAlloc(size);
  initializeMem(srcAddr,size,srcChar);
  //CkPrintf("[%d]Trying to register memory %p\n",idx,srcAddr);
  CmiRegisterMemory((void*)srcAddr,size);
  thisProxy[dest].remoteRMA(size,operation,cb);
}

void commtest::remoteRMA(int len,int op, int cb) {
  size = len;
  operation = 1 - op;
  callb = cb;
  destAddr = (char*)malloc(size*sizeof(char));
  //destAddr = (char*)CmiDMAAlloc(size);
  initializeMem(destAddr,size,srcChar);
  //CkPrintf("[%d]Trying to register memory %p\n",idx,destAddr);
  CmiRegisterMemory((void*)destAddr,size);
  charMsg *cm = new charMsg();
  cm->addr = destAddr;
  thisProxy[dest].recvAddr(cm);
}

void commtest::recvAddr(charMsg *cm)
{
  destAddr = cm->addr;
  //now that we have all info, could do either a get or put
  if(operation==0) {
    CkPrintf("[%d]Trying to do a remote put %p to %p\n",idx,srcAddr,destAddr);
    if(callb==0) {
      pend = CmiPut(idx, dest, (void*)srcAddr, (void*)destAddr, size);
    }
    else {
      void *tmp;
      CmiPutCb(idx, dest, (void*)srcAddr, (void*)destAddr, size, doneOp, tmp);
    }
  }
  else {
    CkPrintf("[%d]Trying to do a remote get %p to %p\n",idx,srcAddr,destAddr);
    if(callb==0) {
      pend = CmiGet(idx, dest, (void*)srcAddr, (void*)destAddr, size);
    }
    else {
      void *tmp;
      CmiGetCb(idx, dest, (void*)srcAddr, (void*)destAddr, size, doneOp, tmp);
    }
  }
  if(callb==0) {
    testDone();
  }
  delete cm;
}

void commtest::verifyCorrectRMA(char c) {
  char *tmp;
  if(idx==0) {
    tmp = srcAddr;
  }
  else {
    tmp = destAddr;
  }
  CkPrintf("[%d]Verifying correct RMA\n",idx);
  bool flag = true;
  for(int i=0; i<size; i++) {
    if(*tmp != c) {
      CkPrintf("[%d]Error: address: %p, expectedValue: %c actualValue: %c\n",idx,tmp,c,*tmp);
      flag = false;
      break;
    }
    tmp++;
  }
  if(!flag) {
    //thisProxy[dest].testDone();
  }
  else {
    CkPrintf("[%d]RMA operation %d correctly finished!!\n",idx,operation);
  }
  mainProxy.done();
}

void commtest::doJnkWork(void) {
  const double left=0.1, right=0.2;
  register double v1=0.1,v2=0.3,v3=1.0;
  int i;
  int iter = 30000000;
  for (i=0;i<iter;i++) {
    v1=(v2+v3)*0.5;
    v2=(v3+v1)*0.5;
    v3=(v1+v2)*0.5;
  }
  CkPrintf("[%d]Did jnk work\n",idx);
  thisProxy[dest].testDone();
}

void commtest::testDone(void) {
  testForCompletion();
  int done = CmiWaitTest(pend);
  CkPrintf("[%d]Test if done %p, %d!!\n",idx,pend,*((int*)pend));
  if(done==1) {
    void *tmp;
    testForCorrectness();
  }
}

void commtest::initializeMem(char *addr, int len, char c) {
  for(int i=0; i<len; i++) {
    *addr = c;
    addr++;
  }
}

void commtest::testForCompletion(void) {
  int done = CmiWaitTest(pend);
  if(done!=1) {
    CkPrintf("[%d]Do jnk work\n",idx);
    thisProxy[dest].doJnkWork();
    done = CmiWaitTest(pend);
  }
}

void commtest::testForCorrectness(void) {
  CkPrintf("[%d]Just finished RMA operation %d\n",idx,operation);
  if(operation==0) { //put operation
    thisProxy[dest].verifyCorrectRMA(srcChar);
  }
  else {
    verifyCorrectRMA(destChar);
  }
}

void doneOp(void *tmp) {
  ((commtest*)(CpvAccess(_cmvar)))->testForCorrectness();
}

#include "onesided.def.h"


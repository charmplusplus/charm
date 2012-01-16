#include <string.h> // for strlen, and strcmp
#include <charm++.h>

#define NITER 1000
#define PAYLOAD 100

#if ! CMK_SMP            /* only test RDMA when non-SMP */

#if defined(CMK_DIRECT) || defined(CMK_USE_IBVERBS)
#define USE_RDMA 1
#endif

#endif

#ifdef USE_RDMA
extern "C" {
#include "cmidirect.h"
}
#endif

class Fancy
{
  char _str[12];
  public:
    Fancy() { _str[0] = '\0'; }
    Fancy(const char *str) {
      strncpy(_str, str, 12);
    }
    int equals(const char *str) const { return !strcmp(str, _str); }
};

class CkArrayIndexFancy : public CkArrayIndex {
  Fancy *f;
  public:
    CkArrayIndexFancy(const char *str) 
    {
        /// Use placement new to ensure that the custom index object is placed in the memory reserved for it in the base class
        f = new (index) Fancy(str);
        nInts=3; 
    }
};

#include "pingpong.decl.h"
class PingMsg : public CMessage_PingMsg
{
  public:
    char *x;

};

class IdMsg : public CMessage_IdMsg
{
  public:
    CkChareID cid;
    IdMsg(CkChareID _cid) : cid(_cid) {}
};

CProxy_main mainProxy;
int iterations;
int payload;

#define P1 0
#define P2 1%CkNumPes()

class main : public CBase_main
{
  int phase;
  CProxy_Ping1 arr1;
  CProxy_Ping2 arr2;
  CProxy_Ping3 arr3;
  CProxy_PingF arrF;
  CProxy_PingC cid;
  CProxy_PingG gid;
  CProxy_PingN ngid;
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg* m)
  {
    if(CkNumPes()>2) {
      CkAbort("Run this program on 1 or 2 processors only.\n");
    }

    iterations=NITER;
    payload=PAYLOAD;
    if(m->argc>1)
      payload=atoi(m->argv[1]);
    if(m->argc>2)
      iterations=atoi(m->argv[2]);
    if(m->argc>3)
      CkPrintf("Usage: pgm +pN [payload] [iterations]\n Where N [1-2], payload (default %d) is integer >0 iterations (default %d) is integer >0 ", PAYLOAD, NITER);
    CkPrintf("Pingpong with payload: %d iterations: %d\n", payload,iterations);
    mainProxy = thishandle;
    phase = 0;
    gid = CProxy_PingG::ckNew();
    ngid = CProxy_PingN::ckNew();
    cid=CProxy_PingC::ckNew(1%CkNumPes());
    cid=CProxy_PingC::ckNew(new IdMsg(cid.ckGetChareID()),0);
    arr1 = CProxy_Ping1::ckNew(2);
    arr2 = CProxy_Ping2::ckNew();
    arr3 = CProxy_Ping3::ckNew();
    arrF = CProxy_PingF::ckNew();
    arr2(0,0).insert(P1);
    arr2(0,1).insert(P2);
    arr2.doneInserting();
    arr3(0,0,0).insert(P1);
    arr3(0,0,1).insert(P2);
    arr3.doneInserting();
    arrF[CkArrayIndexFancy("first")].insert(P1);
    arrF[CkArrayIndexFancy("second")].insert(P2);
    arrF.doneInserting();
    phase=0;
    mainProxy.maindone();
    delete m;
  };

  void maindone(void)
  {
    switch(phase++) {
      case 0:
	arr1[0].start();
	break;
      case 1:
        arr2(0,0).start();
        break;
      case 2:
        arr3(0,0,0).start();
        break;
      case 3:
        arrF[CkArrayIndexFancy("first")].start();
        break;
      case 4:
        cid.start();
        break;
      case 5:
        gid[0].start();
        break;
#ifndef USE_RDMA
      case 6:
        ngid[0].start();
        break;
#else
      case 6:
	  ngid[0].startRDMA();
	  break;
#endif
      default:
        CkExit();
    }
  };
};

class PingG : public CBase_PingG
{
  CProxyElement_PingG *pp;
  int niter;
  int me, nbr;
  double start_time, end_time;
public:
  PingG()
  {
    me = CkMyPe();    
    nbr = (me+1)%CkNumPes();
    pp = new CProxyElement_PingG(thisgroup,nbr);
    niter = 0;
  }
  PingG(CkMigrateMessage *m) {}
  void start(void)
  {
    start_time = CkWallTimer();
    (*pp).recv(new (payload) PingMsg);
  }
  void recv(PingMsg *msg)
  {
    if(me==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        int titer = (CkNumPes()==1)?(iterations/2) : iterations;
        CkPrintf("Roundtrip time for Groups is %lf us\n",
                 1.0e6*(end_time-start_time)/titer);
        delete msg;
        mainProxy.maindone();
      } else {
        (*pp).recv(msg);
      }
    } else {
      (*pp).recv(msg);
    }
  }
};


class PingN : public CBase_PingN
{
  int niter;
  int me, nbr;
#ifdef USE_RDMA 
  struct infiDirectUserHandle shandle,rhandle;
  char *rbuff;
  char *sbuff;
#endif
  double start_time, end_time;
public:
  PingN()
  {
    me = CkMyNode();    
    nbr = (me+1)%CkNumNodes();

    // note: for RMDA in ping you can only have 1 nbr who is both your
    // upstream and downstream which makes this an artificially simple
    // calculation.

    niter = 0;
#ifdef USE_RDMA 
    rbuff=(char *) malloc(payload*sizeof(char));
    sbuff=(char *) malloc(payload*sizeof(char));
    bzero(sbuff,payload);
    // setup persistent comm sender and receiver side
    double OOB=9999999999.0;
    rhandle=CmiDirect_createHandle(nbr,rbuff,payload*sizeof(char),PingN::Wrapper_To_CallBack,(void *) this,OOB);
    thisProxy[nbr].recvHandle((char*) &rhandle,sizeof(struct infiDirectUserHandle));
#endif
  }
  PingN(CkMigrateMessage *m) {}
  void recvHandle(char *ptr,int size)
  {

#ifdef USE_RDMA 
    struct infiDirectUserHandle *_shandle=(struct infiDirectUserHandle *) ptr;
    shandle=*_shandle;
    CmiDirect_assocLocalBuffer(&shandle,sbuff,payload);
#endif
  }
  void start(void)
  {
    start_time = CkWallTimer();
    thisProxy[nbr].recv(new (payload) PingMsg);
  }
  void startRDMA(void)
  {
    niter=0;
    start_time = CkWallTimer();
#ifdef USE_RDMA 
    CmiDirect_put(&shandle);
#else
    CkAbort("do not call startRDMA if you don't actually have RDMA");
#endif
  }

  void recv(PingMsg *msg)
  {
    if(me==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        int titer = (CkNumNodes()==1)?(iterations/2) : iterations;
        CkPrintf("Roundtrip time for NodeGroups is %lf us\n",
                 1.0e6*(end_time-start_time)/titer);
        delete msg;
        mainProxy.maindone();
      } else {
        thisProxy[nbr].recv(msg);
      }
    } else {
      thisProxy[nbr].recv(msg);
    }
  }
  static void Wrapper_To_CallBack(void* pt2Object){
    // explicitly cast to a pointer to PingN
    PingN* mySelf = (PingN*) pt2Object;

    // call member
    if(CkNumNodes() == 0){
      mySelf->recvRDMA();
    }else{
      (mySelf->thisProxy)[CkMyNode()].recvRDMA();   
    }
  }
  // not an entry method, called via Wrapper_To_Callback
  void recvRDMA()
  {
#ifdef USE_RDMA 
    CmiDirect_ready(&rhandle);
#endif
    if(me==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        int titer = (CkNumNodes()==1)?(iterations/2) : iterations;
        CkPrintf("Roundtrip time for NodeGroups RDMA is %lf us\n",
                 1.0e6*(end_time-start_time)/titer);
        mainProxy.maindone();
      } else {
#ifdef USE_RDMA 
	CmiDirect_put(&shandle);
#else
	CkAbort("do not call startRDMA if you don't actually have RDMA");
#endif
      }
    } else {
#ifdef USE_RDMA 
      CmiDirect_put(&shandle);
#else
      CkAbort("do not call startRDMA if you don't actually have RDMA");
#endif
    }
  }

};


class Ping1 : public CBase_Ping1
{
  CProxy_Ping1 *pp;
  int niter;
  double start_time, end_time;
public:
  Ping1()
  {
    pp = new CProxy_Ping1(thisArrayID);
    niter = 0;
  }
  Ping1(CkMigrateMessage *m) {}
  void start(void)
  {
    (*pp)[1].recv(new (payload) PingMsg);
    start_time = CkWallTimer();
  }
  void recv(PingMsg *msg)
  {
    if(thisIndex==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        CkPrintf("Roundtrip time for 1D Arrays is %lf us\n",
                 1.0e6*(end_time-start_time)/iterations);
        //mainProxy.maindone();
        niter=0;
        start_time = CkWallTimer();
        (*pp)[0].trecv(msg);
      } else {
        (*pp)[1].recv(msg);
      }
    } else {
      (*pp)[0].recv(msg);
    }
  }
  void trecv(PingMsg *msg)
  {
    if(thisIndex==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        CkPrintf("Roundtrip time for 1D threaded Arrays is %lf us\n",
                 1.0e6*(end_time-start_time)/iterations);
        mainProxy.maindone();
      } else {
        (*pp)[1].trecv(msg);
      }
    } else {
      (*pp)[0].trecv(msg);
    }
  }
};

class Ping2 : public CBase_Ping2
{
  CProxy_Ping2 *pp;
  int niter;
  double start_time, end_time;
public:
  Ping2()
  {
    pp = new CProxy_Ping2(thisArrayID);
    niter = 0;
  }
  Ping2(CkMigrateMessage *m) {}
  void start(void)
  {
    (*pp)(0,1).recv(new (payload) PingMsg);
    start_time = CkWallTimer();
  }
  void recv(PingMsg *msg)
  {
    if(thisIndex.y==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        CkPrintf("Roundtrip time for 2D Arrays is %lf us\n",
                 1.0e6*(end_time-start_time)/iterations);
        mainProxy.maindone();
      } else {
        (*pp)(0,1).recv(msg);
      }
    } else {
      (*pp)(0,0).recv(msg);
    }
  }
};

class Ping3 : public CBase_Ping3
{
  CProxy_Ping3 *pp;
  int niter;
  double start_time, end_time;
public:
  Ping3()
  {
    pp = new CProxy_Ping3(thisArrayID);
    niter = 0;
  }
  Ping3(CkMigrateMessage *m) {}
  void start(void)
  {
    (*pp)(0,0,1).recv(new (payload) PingMsg);
    start_time = CkWallTimer();
  }
  void recv(PingMsg *msg)
  {
    if(thisIndex.z==0) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        CkPrintf("Roundtrip time for 3D Arrays is %lf us\n",
                 1.0e6*(end_time-start_time)/iterations);
        mainProxy.maindone();
      } else {
        (*pp)(0,0,1).recv(msg);
      }
    } else {
      (*pp)(0,0,0).recv(msg);
    }
  }
};

class PingF : public CBase_PingF
{
  CProxy_PingF *pp;
  int niter;
  double start_time, end_time;
  int first;
public:
  PingF()
  {
    pp = new CProxy_PingF(thisArrayID);
    niter = 0;
    first = thisIndex.equals("first") ? 1 : 0;
  }
  PingF(CkMigrateMessage *m) {}
  void start(void)
  {
    (*pp)[CkArrayIndexFancy("second")].recv(new (payload) PingMsg);
    start_time = CkWallTimer();
  }
  void recv(PingMsg *msg)
  {
    CkArrayIndexFancy partner((char *)(first?"second" : "first"));
    if(first) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        CkPrintf("Roundtrip time for Fancy Arrays is %lf us\n",
                 1.0e6*(end_time-start_time)/iterations);
	delete msg;
        mainProxy.maindone();
      } else {
        (*pp)[partner].recv(msg);
      }
    } else {
      (*pp)[partner].recv(msg);
    }
  }
};

class PingC : public CBase_PingC
{
  CProxy_PingC *pp;
  int niter;
  double start_time, end_time;
  int first;
 public:
  PingC(void)
  {
    first = 0;
  }
  PingC(IdMsg *msg)
  {
    first = 1;
    CProxy_PingC pc(msg->cid);
    msg->cid = thishandle;
    pc.exchange(msg);
  }
  PingC(CkMigrateMessage *m) {}
  void start(void)
  {
    niter = 0;
    pp->recvReuse(new (payload) PingMsg);
    start_time = CkWallTimer();
  }
  void exchange(IdMsg *msg)
  {
    if(first) {
      pp = new CProxy_PingC(msg->cid);
      delete msg;
    } else {
      pp = new CProxy_PingC(msg->cid);
      msg->cid = thishandle;
      pp->exchange(msg);
    }
  }
  void recvReuse(PingMsg *msg)
  {
    if(first) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        CkPrintf("Roundtrip time for Chares (reuse msgs) is %lf us\n",
                 1.0e6*(end_time-start_time)/iterations);
        niter = 0;
        delete msg;
        pp->recv(new (payload) PingMsg);
        start_time = CkWallTimer();
      } else {
        pp->recvReuse(msg);
      }
    } else {
      pp->recvReuse(msg);
    }
  }
  void recv(PingMsg *msg)
  {
    delete msg;
    if(first) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        CkPrintf("Roundtrip time for Chares (new/del msgs) is %lf us\n",
                 1.0e6*(end_time-start_time)/iterations);
        niter = 0;
        pp->trecv(new (payload) PingMsg);
        start_time = CkWallTimer();
      } else {
        pp->recv(new (payload) PingMsg);
      }
    } else {
      pp->recv(new (payload) PingMsg);
    }
  }
  void trecv(PingMsg *msg)
  {
    if(first) {
      niter++;
      if(niter==iterations) {
        end_time = CkWallTimer();
        CkPrintf("Roundtrip time for threaded Chares (reuse) is %lf us\n",
                 1.0e6*(end_time-start_time)/iterations);
	delete msg;
        mainProxy.maindone();
      } else {
        pp->trecv(msg);
      }
    } else {
      pp->trecv(msg);
    }
  }
};
#include "pingpong.def.h"

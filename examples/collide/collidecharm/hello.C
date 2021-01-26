/*
   Simple Charm++ collision detection test program--
   Orion Sky Lawlor, olawlor@acm.org, 2003/3/18
   */
#include <stdio.h>
#include <string.h>
#include "collidecharm.h"
#include "hello.decl.h"

CProxy_main mid;
CProxy_Hello arr;
int nElements;

void printCollisionHandler(void *param,int nColl,Collision *colls)
{
  CkPrintf("**********************************************\n");
  int nPrint=nColl;
  const int maxPrint=30;
  if (nPrint>maxPrint) nPrint=maxPrint;
  for (int c=0;c<nPrint;c++) {
    CkPrintf("%d:%d hits %d:%d\n",
        colls[c].A.chunk,colls[c].A.number,
        colls[c].B.chunk,colls[c].B.number);
  }
  CkPrintf("**********************************************\n");
  mid.maindone();
}

class main : public CBase_main
{
  public:
    main(CkMigrateMessage *m) {}
    main(CkArgMsg* m)
    {
      nElements=5;
      if(m->argc > 1) nElements = atoi(m->argv[1]);
      delete m;
      CkPrintf("Running Hello on %d processors for %d elements\n",
          CkNumPes(),nElements);
      mid = thishandle;

      CollideGrid3d gridMap(CkVector3d(0,0,0),CkVector3d(2,100,2));

#if COLLIDE_USE_DIST
      CProxy_collResultCollector collProxy = CProxy_collResultCollector::ckNew();
      CkStartQD(CkCallback(CkIndex_main::maindone(), thisProxy));
      CollideHandle collide = CollideCreate(gridMap,
            CollideDistributedClient(CkCallback(CkIndex_collResultCollector::myColls(NULL), collProxy)));
#else
#if COLLIDE_USE_CB
      CollideHandle collide = CollideCreate(gridMap,
          CollideSerialClient(CkCallback(CkIndex_main::printCollisionCb(NULL), thisProxy)));
#else
      CollideHandle collide=CollideCreate(gridMap,
          CollideSerialClient(printCollisionHandler,0));
#endif
#endif

      arr = CProxy_Hello::ckNew(collide,nElements);

      arr.DoIt();
    };

    void maindone(void)
    {
      CkPrintf("All done\n");
      CkExit();
    };

    void printCollisionCb(CkReductionMsg *msg) {
      Collision *colls = (Collision *)msg->getData();
      int nColl = msg->getSize()/sizeof(Collision);
      CkPrintf("Calling collision handler using a callback\n");
      printCollisionHandler(NULL, nColl, colls);

      delete msg;
    }
};

class collResultCollector : public CBase_collResultCollector {
  public:
  collResultCollector() {}

  void myColls(CkDataMsg *msg) {
    Collision *colls = (Collision *)msg->getData();
    int nColl = msg->getSize()/sizeof(Collision);
    int nPrint=nColl;
    const int maxPrint=30;
    if (nPrint>maxPrint) nPrint=maxPrint;
    for (int c=0;c<nPrint;c++) {
      CkPrintf("%d:%d hits %d:%d\n",
          colls[c].A.chunk,colls[c].A.number,
          colls[c].B.chunk,colls[c].B.number);
    }
    delete msg;
  }
};

class Hello : public CBase_Hello
{
  CollideHandle collide;
  int nTimes;
  public:
  Hello(const CollideHandle &collide_) :collide(collide_)
  {
    CkPrintf("Creating element %d on PE %d\n",thisIndex,CkMyPe());
    nTimes=0;
    CollideRegister(collide,thisIndex);
  }

  Hello(CkMigrateMessage *m) : CBase_Hello(m) {}
  void pup(PUP::er &p) {
    p|collide;
    if (p.isUnpacking())
      CollideRegister(collide,thisIndex);
  }
  ~Hello() {
    CollideUnregister(collide,thisIndex);
  }

  void DoIt(void)
  {
    CkPrintf("Contributing to reduction %d, element %04d\n",nTimes,thisIndex);
    CkVector3d o(-6.8,7.9,8.0), x(4.0,0,0), y(0,0.3,0);
    CkVector3d boxSize(0.2,0.2,0.2);
    int nBoxes=1000;
    bbox3d *box=new bbox3d[nBoxes];
    for (int i=0;i<nBoxes;i++) {
      CkVector3d c(o+x*thisIndex+y*i);
      CkVector3d c2(c+boxSize);
      box[i].empty();
      box[i].add(c); box[i].add(c2);
    }
    // first box stretches over into next object:
    box[0].add(o+x*(thisIndex+1.5)+y*2);

    CollideBoxesPrio(collide,thisIndex,nBoxes,box,NULL);

    delete[] box;
    nTimes++;
  }
};

#include "hello.def.h"


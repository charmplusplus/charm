#include <stdio.h>
#include <string.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#include "charm++.h"
#include "ComlibManager.h"
#include "EachToManyMulticastStrategy.h"
#include "StreamingStrategy.h"
#include "DummyStrategy.h"
#include "bench.decl.h"

#define USELIB  1
#define MAXITER 5000
#define NUMPASS 2

int fraction = 1;  /* readonly */
/*readonly*/ CkChareID mid;
/*readonly*/ CProxy_Bench arr;
/*readonly*/ int nElements;

class BenchMessage : public CMessage_BenchMessage {
public:
    char *data;

    static void *alloc(int mnum, size_t size, int *sizes, int priobits){
        int total_size = size + CK_ALIGN(sizeof(char) * sizes[0], 8);
        BenchMessage *dmsg = (BenchMessage *)CkAllocMsg(mnum, total_size, 
                                                        priobits);
	dmsg->data = (char *)dmsg + sizeof(BenchMessage);
        return (void *)dmsg;	
    }
    
    static void *pack(BenchMessage *msg){
        return (void *)msg;
    }
    
    static BenchMessage *unpack(void *buf){
        BenchMessage *bmsg = (BenchMessage *)buf;
	bmsg->data = (char *)bmsg + sizeof(BenchMessage);
        return bmsg;
    }
};

void reductionClient(void *param, int dataSize, void *data){
    arr.start(0);
}

/*mainchare*/
class Main : public Chare
{
    int pass, superpass;
    double curTime;
    int mcount;
    int size;

public:
    Main(CkArgMsg* m)
    {
        int stratID = 0;
        //Process command-line arguments
	pass = 0;
	superpass = 0;
        nElements = CkNumPes();

        mcount = 0;

        size = 128;
        if(m->argc > 1 ) size = atoi(m->argv[1]);
	if(m->argc > 2 ) //fraction=atoi(m->argv[2]);
            nElements = atoi(m->argv[2]);
        //delete m;
        
        //Start the computation
        CkPrintf("Running Bench on %d processors for %d elements with %d byte messages\n", CkNumPes(), nElements, size);
        
        mid = thishandle;        
        //ComlibInstanceHandle tmpInstance = CkGetComlibInstance();
        ComlibInstanceHandle cinst = CkGetComlibInstance();
	
        arr = CProxy_Bench::ckNew();

	DummyStrategy *dstrat = new DummyStrategy();
        EachToManyMulticastStrategy *strat = new EachToManyMulticastStrategy(USE_MESH, arr.ckGetArrayID(), arr.ckGetArrayID());
        
        StreamingStrategy *sstrat = new StreamingStrategy(10, 10);
        //sstrat->enableShortArrayMessagePacking();

        cinst.setStrategy(strat);                
        //tmpInstance.setStrategy(sstrat);
        //cinst = tmpInstance;

        for(int count =0; count < nElements; count++)
	  arr[count].insert(cinst);

        arr.doneInserting();
        arr.setReductionClient(reductionClient, NULL);

	curTime = CkWallTimer();
        arr.start(size);
    };
    
    void send(void) {
        
      mcount ++;
      
      //printf("Count = %d\n", count);
      
      if (mcount == nElements){
	
	pass ++;
	mcount = 0;
	
	CkPrintf("%d %5.4lf\n", size, (CmiWallTimer() - curTime)*1000/MAXITER);
	curTime = CkWallTimer();
	
	if(pass == NUMPASS)
	  done();
	else            
	  arr.start(size);
      }
    }
    
    void done()
    {	
      superpass ++;
      mcount = 0;
      pass = 0;
      
      if(superpass == 20)
	CkExit();
      else {
	if(superpass < 20)
	  size += 50;
	else if(superpass < 30)
	  size += 100;
	else if(superpass < 40)
	  size += 200;
	else if(superpass < 50)
	  size += 500;
	
	arr.start(size);
      }
    }
};

/*array [1D]*/
class Bench : public ArrayElement1D
{
  int pass;
    int mcount;
    int ite;
    int msize;
    double startTime;
    ComlibInstanceHandle myInst;
    CProxy_Bench arrd;      

public:
  
    Bench(ComlibInstanceHandle cinst)
    {   
        pass = 0;
        mcount = 0;
        ite = 0;
	msize = 0;

        myInst = cinst;

        myInst.setSourcePe();

        usesAtSync = CmiTrue;
        setMigratable(true);

        arrd = arr;
        ComlibDelegateProxy(&arrd);
    }
    
    Bench(CkMigrateMessage *m) {
        CkPrintf("Migrated to %d\n", CkMyPe());
        //myInst = cinst;
    }
    
    void sendMessage()
    {
#ifdef USELIB
        myInst.beginIteration();
#endif        
        //for(int count = 0; count < nElements; count ++){
	for(int dest = thisIndex + 1; dest < thisIndex + nElements/fraction;
	    dest ++){
            
	    int count = dest % nElements;
            if(count == thisIndex)
                continue;
            
            //CkPrintf("[%d] Sending Message from %d to %d\n", CkMyPe(), thisIndex, count);

#ifdef USELIB
            arrd[count].receiveMessage(new (&msize, 0) BenchMessage); 
#else
	    arr[count].receiveMessage(new (&msize, 0) BenchMessage);
#endif
        }

#ifdef USELIB
        myInst.endIteration();
#endif

        //CkPrintf("After SendMessage %d\n", thisIndex);
    }
    
    void receiveMessage(BenchMessage *bmsg){
        
        delete bmsg;
        mcount ++;
        
        ComlibPrintf("In Receive Message %d %d %d\n", thisIndex, CkMyPe(), pass);

        if(mcount == nElements/fraction - 1){
            mcount = 0;            
            pass ++;            
            CProxy_Main mainProxy(mid);
            if(pass == MAXITER){
		pass = 0;

		mainProxy.send();
            }
            else {
                sendMessage();
                //int x = 0;
                //contribute(sizeof(int), (void *)&x, CkReduction::sum_int);
            }
        }
    }

    void start(int messagesize){
        msize = messagesize;
	/*
	if(ite % NUMPASS == NUMPASS/2 || ite % NUMPASS == 1)  
	  //Call atsync in the middle and in the end
	  AtSync();
        else
	*/
	  sendMessage();
        
        //CkPrintf("In Start\n");
        ite ++;
    }

    void ResumeFromSync() {
        //CkPrintf("%d: resuming\n", CkMyPe());
        myInst.setSourcePe();
	sendMessage();
    }

    void pup(PUP::er &p) {
        if(p.isPacking())
            CkPrintf("Migrating from %d\n", CkMyPe());

        ArrayElement1D::pup(p);
        p | pass ;
        p | mcount ;
        p | ite ;
        p | startTime;
        p | arrd;
        p | myInst;
	p | msize;
    }
};


#include "bench.def.h"


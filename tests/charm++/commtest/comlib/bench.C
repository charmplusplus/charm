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
#define MAXPASS 100
#define MESSAGESIZE CkpvAccess(msgSize)

int fraction = 1;  /* readonly */
/*readonly*/ CkChareID mid;
/*readonly*/ CProxy_Bench arr;
/*readonly*/ int nElements;

CkpvDeclare(int, msgSize);

class BenchMessage : public CMessage_BenchMessage {
public:
    int size;
    char *data;
    //int src;
    
    static void *alloc(int mnum, size_t size, int *sizes, int priobits){
        int total_size = size + sizeof(char) * sizes[0];
        BenchMessage *dmsg = (BenchMessage *)CkAllocMsg(mnum, total_size, 
                                                        priobits);
        dmsg->size = sizes[0];
        
        return (void *)dmsg;
    }
    
    static void *pack(BenchMessage *msg){
        return (void *)msg;
    }
    
    static BenchMessage *unpack(void *buf){
        BenchMessage *bmsg = (BenchMessage *)buf;
        return bmsg;
    }
};

void reductionClient(void *param, int dataSize, void *data){
    arr.start(MESSAGESIZE);
}

/*mainchare*/
class Main : public Chare
{
    int pass, superpass;
    double curTime;
    int mcount;
    
public:
    Main(CkArgMsg* m)
    {
        int stratID = 0;
        //Process command-line arguments
	pass = 0;
	superpass = 0;
        nElements = CkNumPes();

        mcount = 0;

        CkpvInitialize(int, msgSize); 
        
        MESSAGESIZE = 128;
        if(m->argc > 1 ) MESSAGESIZE=atoi(m->argv[1]);
	if(m->argc > 2 ) //fraction=atoi(m->argv[2]);
            nElements = atoi(m->argv[2]);
        //delete m;
        
        //Start the computation
        CkPrintf("Running Bench on %d processors for %d elements with %d byte messages\n", CkNumPes(), nElements, MESSAGESIZE);
        
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
        arr.start(MESSAGESIZE);
    };
    
    void send(void) {
        
        mcount ++;
        
        //printf("Count = %d\n", count);
        
        if (mcount == nElements){
            
            pass ++;
            mcount = 0;
            
            arr.start(MESSAGESIZE);
        }
    }
    
    void done()
    {
	
        mcount ++;
        
        if(mcount == nElements) {
            
            CkPrintf("%d %5.4lf\n", MESSAGESIZE, (CmiWallTimer() - curTime)*1000/MAXPASS);

	    curTime = CkWallTimer();
	    superpass ++;
	    pass = 0;
            mcount = 0;

	    if(superpass == 20)
		CkExit();
	    else {
	      if(superpass < 20)
		  MESSAGESIZE += 50;
	      else if(superpass < 30)
		  MESSAGESIZE += 100;
	      else if(superpass < 40)
		  MESSAGESIZE += 200;
	      else if(superpass < 50)
		  MESSAGESIZE += 500;
	      
	      arr.start(MESSAGESIZE);
	    }
        }
    }
};

/*array [1D]*/
class Bench : public ArrayElement1D
{
    int pass;
    int mcount;
    int ite;
    double startTime;
    int firstEntryFlag, sendFinishedFlag, doneFlag;
    ComlibInstanceHandle myInst;
    CProxy_Bench arrd;      

public:
  
    Bench(ComlibInstanceHandle cinst)
    {   
        CkpvInitialize(int, msgSize); 
        pass = 0;
        mcount = 0;
        ite = 0;

        firstEntryFlag = 0;
        sendFinishedFlag = 0;
        doneFlag = 0;
        myInst = cinst;

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
        for(int count = 0; count < nElements; count ++){
            //for(int dest = thisIndex + 1; dest < thisIndex + nElements/fraction;
            // dest ++){
            
            //int count = dest % nElements;
            if(count == thisIndex)
                continue;
            
            //CkPrintf("[%d] Sending Message from %d to %d\n", CkMyPe(), thisIndex, count);

            int size = MESSAGESIZE;
#ifdef USELIB
            arrd[count].receiveMessage(new (&size, 0) BenchMessage); 
#else
	    arr[count].receiveMessage(new (&size, 0) BenchMessage);
#endif
        }

#ifdef USELIB
        myInst.endIteration();
#endif

        //CkPrintf("After SendMessage %d\n", thisIndex);

        sendFinishedFlag = 1;
    }
    
    void receiveMessage(BenchMessage *bmsg){
        
        if(!firstEntryFlag) {
            startTime = CkWallTimer();
            firstEntryFlag = 1;
        }
        
        delete bmsg;
        mcount ++;
        
        ComlibPrintf("In Receive Message %d %d %d\n", thisIndex, CkMyPe(), pass);

        if((mcount == nElements/fraction - 1) /*&& (sendFinishedFlag)*/){
            mcount = 0;            
            pass ++;            
            CProxy_Main mainProxy(mid);
            if(pass == MAXPASS){
		pass = 0;

                //AtSync();
		mainProxy.done();
            }
            else {
                sendMessage();
                //int x = 0;
                //contribute(sizeof(int), (void *)&x, CkReduction::sum_int);
            }

            firstEntryFlag = 0;
            sendFinishedFlag = 0;
            doneFlag = 0;
        }
        else doneFlag = 1;
    }

    void start(int messagesize){
        MESSAGESIZE = messagesize;
        //MESSAGESIZE = 128;

        //if(ite == 0)
        sendMessage();
        //else {
        //  CkPrintf("Calling AtSync()\n");
        //    AtSync();
        //}
        
        //CkPrintf("In Start\n");
        ite ++;
    }

    void ResumeFromSync() {
        CkPrintf("%d: resuming\n", CkMyPe());
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
        p | firstEntryFlag ;
        p | sendFinishedFlag ;
        p | doneFlag ;
        p | arrd;
        p | myInst;
    }
};


#include "bench.def.h"


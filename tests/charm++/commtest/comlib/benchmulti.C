
#include <stdio.h>
#include <string.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#include "ComlibManager.h"
#include "EachToManyMulticastStrategy.h"
#include "RingMulticastStrategy.h"
#include "BroadcastStrategy.h"
#include "bench.decl.h"

#define USELIB
#define MAXPASS 100

int MESSAGESIZE=128;
int fraction = 1;
/*readonly*/ CkChareID mid;
/*readonly*/ CProxy_Bench arr;
/*readonly*/ int nElements;

void callbackhandler(void *message){
    //CkPrintf("[%d]In callback function\n", CkMyPe());
    
    BenchMessage *bm = (BenchMessage *)EnvToUsr((envelope *)message);
    arr[CkMyPe()].receiveMessage(bm);
}

class BenchMessage : public CkMcastBaseMsg, public CMessage_BenchMessage {
public:
    int size;
    char *data;
    int src;
    
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
    arr.sendMessage();
}

/*mainchare*/
class Main : public Chare
{
    int pass, superpass;
    double curTime;
public:
    Main(CkArgMsg* m)
    {
        //Process command-line arguments
	pass = 0;
	superpass = 0;
        nElements= CkNumPes();

        if(m->argc > 1 ) MESSAGESIZE=atoi(m->argv[1]);
	//if(m->argc > 2 ) fraction=atoi(m->argv[2]);

        if(m->argc > 2 ) nElements=atoi(m->argv[2]);

        delete m;
        
        CkPrintf("Running Bench on %d processors for %d elements with %d byte messages\n", CkNumPes(), nElements, MESSAGESIZE);
        
        mid = thishandle;        

	arr = CProxy_Bench::ckNew();
	arr.setReductionClient(reductionClient, NULL);
                
        int count = 0;
        CkArrayIndexMax *elem_array = new CkArrayIndexMax[nElements/fraction];
        for(count = 0; count < nElements/fraction; count ++) {
            elem_array[count] = CkArrayIndex1D(count);
        }

        EachToManyMulticastStrategy *strat = new 
            EachToManyMulticastStrategy(USE_MESH, arr.ckGetArrayID(), 
                                        arr.ckGetArrayID(), 
                                        nElements/fraction, elem_array,
                                        nElements/fraction, elem_array); 
        
        DirectMulticastStrategy *dstrat = new DirectMulticastStrategy
            (arr.ckGetArrayID());

	RingMulticastStrategy *rstrat = new RingMulticastStrategy
            (arr.ckGetArrayID(), arr.ckGetArrayID());

        BroadcastStrategy *bstrat = new BroadcastStrategy(arr.ckGetArrayID(),
                                                          USE_HYPERCUBE);
        
        CkPrintf("After creating array\n");
	CkArrayID aid = arr.ckGetArrayID();

        ComlibInstanceHandle cinst = CkGetComlibInstance();        
        cinst.setStrategy(bstrat);
        ComlibPrintf("After register strategy\n");

        for(count = 0; count < nElements; count++)
            arr[count].insert(cinst);
        arr.doneInserting();
        
	curTime = CkWallTimer();
	arr.sendMessage();
        CkPrintf("After Main\n");
    };
    
    void send(void) {
        
        static int count = 0;
        count ++;

        if (count == nElements){
            pass ++;
            count = 0;
	    arr.sendMessage();
        }
    }
    
    void done()
    {
        static int count = 0;
        
        count ++;
        
        if(count == nElements) {
            CkPrintf("%d %5.4lf\n", MESSAGESIZE, (CmiWallTimer() - curTime)*1000/MAXPASS);

	    curTime = CkWallTimer();
	    superpass ++;
	    pass = 0;
            count = 0;

	    if(superpass == 50)
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
class Bench : public CBase_Bench //ArrayElement1D
{
    int pass;
    int mcount;
    double time, startTime;
    int firstEntryFlag, sendFinishedFlag;
    CProxySection_Bench sproxy;
    CProxy_Bench arrd;
    ComlibInstanceHandle myinst;

public:

    void pup(PUP::er &p) {
        if(p.isPacking())
            CkPrintf("Migrating from %d\n", CkMyPe());

        ArrayElement1D::pup(p);
        p | pass ;
        p | mcount ;
        p | time;
        p | startTime;
        p | firstEntryFlag ;
        p | sendFinishedFlag ;
        p | sproxy ;
        p | myinst;
        p | arrd;
    }
  
    Bench(ComlibInstanceHandle cinst)
    {   
        pass = 0;
        mcount = 0;
        time = 0.0;
        
        firstEntryFlag = 0;
        sendFinishedFlag = 0;

        arrd = thisProxy;
        ComlibDelegateProxy(&arrd);

        CkArrayIndexMax *elem_array = new CkArrayIndexMax[nElements/fraction];
        for(int count = 0; count < nElements/fraction; count ++) 
            elem_array[count] = CkArrayIndex1D(count);
        
        sproxy = CProxySection_Bench::ckNew
            (thisProxy.ckGetArrayID(), elem_array, nElements/fraction); 
        //ComlibInitSection(sproxy.ckGetSectionInfo());
        ComlibDelegateProxy(&sproxy);

        usesAtSync = CmiTrue;
        setMigratable(true);
        myinst = cinst;
    }
    
    Bench(CkMigrateMessage *m) {
        CkPrintf(" Migrated to %d\n", CkMyPe());
    }
    
    void sendMessage()
    {
        if(thisIndex >= nElements/fraction) {
            finishPass();
            return;
        }
	
#ifdef USELIB
        myinst.beginIteration();
#endif

	int count = 0;
	int size = MESSAGESIZE;
	
#ifdef USELIB
        //ComlibDelegateProxy(&arrd);
        arrd.receiveMessage(new(&size, 0) BenchMessage);
        //sproxy.receiveMessage(new(&size, 0) BenchMessage);
#else
	arr[count].receiveMessage(new (&size, 0) BenchMessage);
#endif

#ifdef USELIB
        myinst.endIteration();
#endif

        sendFinishedFlag = 1;	
    }
    
    void receiveMessage(BenchMessage *bmsg){
        
        ComlibPrintf("[%d][%d] In Receive Message \n", CkMyPe(), thisIndex);
        
        if(!firstEntryFlag) {
            startTime = CkWallTimer();
            firstEntryFlag = 1;
        }
        
        delete bmsg;
        
        mcount ++;

        if((mcount == nElements/fraction) /*&& (sendFinishedFlag)*/){
            finishPass();
        }
    }

    void start(int messagesize){
        //CkPrintf("In Start\n");
	MESSAGESIZE = 128;  //messagesize;
        
        //if(firstEntryFlag) {
        //  CkPrintf("Calling AtSync\n");
        //  AtSync();
        //}
        //else
        sendMessage();
    }

    void ResumeFromSync() {
        ComlibResetSectionProxy(&sproxy);
        myinst.setSourcePe();
        sendMessage();
    }

    void finishPass(){
        mcount = 0;            
        pass ++;        
        time += CkWallTimer() - startTime;

        CProxy_Main mainProxy(mid);
        if(pass == MAXPASS){
            pass = 0;
            mainProxy.done();
        }
        else {
            sendMessage();
            int x = 0;
            //contribute(sizeof(int), (void *)&x, CkReduction::sum_int);
        }
        
        sendFinishedFlag = 0;
    }
};

#include "bench.def.h"

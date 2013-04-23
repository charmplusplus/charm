
/*******************************************
        This benchmark tests all to all multicast in Charm++ using the
        communication library.

         To Run
             benchmulti <message_size> <array_elements>

         Defaults
           message size = 128b
           array elements = CkNumPes

         Performance tips
            - Use a maxpass of 1000
            - Use +LBOff to remove loadbalancing overheads

Sameer Kumar 10/28/04      
**************************************************/

#include <stdio.h>
#include <string.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#include "ComlibManager.h"
#include "EachToManyMulticastStrategy.h"
#include "BroadcastStrategy.h"
#include "bench.decl.h"

#define USELIB
#define MAXPASS 10

int MESSAGESIZE=128;
/*readonly*/ CkChareID mid;
/*readonly*/ CProxy_Bench arr;
/*readonly*/ int nElements;

//Old stupid way of creating messages
class BenchMessage : public CMessage_BenchMessage {
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

/*mainchare*/
class Main : public CBase_Main
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

        if(m->argc > 2 ) nElements=atoi(m->argv[2]);

        delete m;
        
        CkPrintf("Running Bench on %d processors for %d elements with %d byte messages\n", CkNumPes(), nElements, MESSAGESIZE);
        
        mid = thishandle;        

	arr = CProxy_Bench::ckNew();
                
        int count = 0;
        CkArrayIndex *elem_array = new CkArrayIndex[nElements];
        for(count = 0; count < nElements; count ++) {
            elem_array[count] = CkArrayIndex1D(count);
        }

        //Create strategy
        EachToManyMulticastStrategy *strat = new 
            EachToManyMulticastStrategy(USE_MESH, arr.ckGetArrayID(), 
                                        arr.ckGetArrayID(),
                                        nElements, elem_array,
                                        nElements, elem_array);

        //Use the multicast learner
        strat->setMulticast();
        strat->enableLearning();

        //Alltoall multicast is effectively an all-to-all broadcast.
        //So we can try the broadcast strategy here too
        BroadcastStrategy *bstrat = new BroadcastStrategy
            (arr.ckGetArrayID(), USE_HYPERCUBE);
        
        //CkPrintf("After creating array\n");
	CkArrayID aid = arr.ckGetArrayID();

        ComlibInstanceHandle cinst = CkGetComlibInstance();        
        cinst.setStrategy(strat);
        ComlibPrintf("After register strategy\n");

        for(count = 0; count < nElements; count++)
            arr[count].insert(cinst);
        arr.doneInserting();
        
	curTime = CkWallTimer();
	arr.sendMessage();
        //        CkPrintf("After Main\n");
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
    
    //Finished a phase, increase message size and start next phase
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

	    if(superpass == 10)
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


//Charm++ array to do all-to-all multicast
/*array [1D]*/
class Bench : public CBase_Bench 
{
    int pass;
    int mcount;
    double time, startTime;
    int firstEntryFlag, sendFinishedFlag;
    CProxy_Bench arrd;
    ComlibInstanceHandle myinst;

public:

    //A pup to migrate elements, because we always need one 
    void pup(PUP::er &p) {
        //if(p.isPacking())
        //  CkPrintf("Migrating from %d\n", CkMyPe());

        CBase_Bench::pup(p);
        p | pass ;
        p | mcount ;
        p | time;
        p | startTime;
        p | firstEntryFlag ;
        p | sendFinishedFlag ;
        p | myinst;
        p | arrd;
    }
  
    //Constructor
    Bench(ComlibInstanceHandle cinst)
    {   
        pass = 0;
        mcount = 0;
        time = 0.0;
        
        firstEntryFlag = 0;
        sendFinishedFlag = 0;

        arrd = thisProxy;
        ComlibDelegateProxy(&arrd);

        CkArrayIndex *elem_array = new CkArrayIndex[nElements];
        for(int count = 0; count < nElements; count ++) 
            elem_array[count] = CkArrayIndex1D(count);
        
        usesAtSync = true;
        setMigratable(true);
        myinst = cinst;
    }
    
    //Migrate constrctor
    Bench(CkMigrateMessage *m) {
        //        CkPrintf(" Migrated to %d\n", CkMyPe());
    }
    
    //Send the multicast message, notice only one is sent.
    void sendMessage()
    {
        if(thisIndex >= nElements) {
            finishPass();
            return;
        }
	
#ifdef USELIB
        myinst.beginIteration();
#endif

	int count = 0;
	int size = MESSAGESIZE;
	
#ifdef USELIB
        arrd.receiveMessage(new(&size, 0) BenchMessage);
#else
	arr[count].receiveMessage(new (&size, 0) BenchMessage);
#endif

#ifdef USELIB
        myinst.endIteration();
#endif

        sendFinishedFlag = 1;	
    }

    //Receive messages. Once all are received, initiate next step with
    //larger message size
    void receiveMessage(BenchMessage *bmsg){
        
        ComlibPrintf("[%d][%d] In Receive Message \n", CkMyPe(), thisIndex);
        
        if(!firstEntryFlag) {
            startTime = CkWallTimer();
            firstEntryFlag = 1;
        }
        
        delete bmsg;
        
        mcount ++;

        if(mcount == nElements){
            finishPass();
        }
    }

    void start(int messagesize){
        //CkPrintf("In Start\n");
	MESSAGESIZE = messagesize;
        
        if(firstEntryFlag) {
            //            CkPrintf("Calling AtSync\n");
            AtSync();
        }
        else
            sendMessage();
    }

    //Resume from loadbalancing
    void ResumeFromSync() {
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
        }
        
        sendFinishedFlag = 0;
    }
};

#include "bench.def.h"

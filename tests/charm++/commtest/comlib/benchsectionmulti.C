
/*******************************************
        This benchmark tests section multicast feature of the
        Communication library with a many to many multicast benchmark.
        Section multicasts are traditionally used for one-to-many
        multicast operations, but this benchmark tests them in an
        many-to-many fashion.

         To Run
             benchsectionmulti <message_size> <array_elements>

         Defaults
           message size = 128b
           array elements = CkNumPes

         Performance tips
            - Use a maxpass of 1000
            - Use +LBOff to remove loadbalancing statistics 
              collection overheads

Sameer Kumar 03/17/05      
**************************************************/

#include <stdio.h>
#include <string.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#include "ComlibManager.h"
#include "RingMulticastStrategy.h"
#include "DirectMulticastStrategy.h"
#include "bench.decl.h"

#define USELIB
#define MAXPASS 10

int MESSAGESIZE=128;
/*readonly*/ CkChareID mid;
/*readonly*/ CProxy_Bench arr;
/*readonly*/ int nElements;

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

        //Different section multicast strategies
        DirectMulticastStrategy *dstrat = new DirectMulticastStrategy
            (arr.ckGetArrayID());

	RingMulticastStrategy *rstrat = new RingMulticastStrategy(arr.ckGetArrayID());

        //        CkPrintf("After creating array\n");
	CkArrayID aid = arr.ckGetArrayID();

        ComlibInstanceHandle cinst = CkGetComlibInstance();        
        cinst.setStrategy(dstrat);
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
    
    //Phase done, initiate next phase with larger message sizes
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

/*array [1D]*/
class Bench : public CBase_Bench
{
    int pass;
    int mcount;
    double time, startTime;
    int firstEntryFlag, sendFinishedFlag;

    CProxySection_Bench sproxy;  //Section proxy, which decides who
                                 //all this array element is sending
                                 //to
    ComlibInstanceHandle myinst;

public:

    void pup(PUP::er &p) {
        //if(p.isPacking())
        //            CkPrintf("Migrating from %d\n", CkMyPe());

        ArrayElement1D::pup(p);
        p | pass ;
        p | mcount ;
        p | time;
        p | startTime;
        p | firstEntryFlag ;
        p | sendFinishedFlag ;
        p | sproxy ;
        p | myinst;
    }
  
    Bench(ComlibInstanceHandle cinst)
    {   
        pass = 0;
        mcount = 0;
        time = 0.0;
        
        firstEntryFlag = 0;
        sendFinishedFlag = 0;

        CkArrayIndex *elem_array = new CkArrayIndex[nElements];
        for(int count = 0; count < nElements; count ++) 
            elem_array[count] = CkArrayIndex1D(count);
        
        //Create the section proxy on all array elements
        //Later subset test should be added to this benchmark 
        sproxy = CProxySection_Bench::ckNew
            (thisProxy.ckGetArrayID(), elem_array, nElements); 
        ComlibResetSectionProxy(&sproxy);
        ComlibDelegateProxy(&sproxy);

        usesAtSync = CmiTrue;
        setMigratable(true);
        myinst = cinst;
    }
    
    Bench(CkMigrateMessage *m) {
        //CkPrintf(" Migrated to %d\n", CkMyPe());
    }

    //Send messages through the section proxy
    void sendMessage()
    {
        if(thisIndex >= nElements) {
            finishPass();
            return;
        }
	
	int count = 0;
	int size = MESSAGESIZE;
	
#ifdef USELIB
        BenchMessage *bmsg = new(&size, 0) BenchMessage;
        bmsg->src = thisIndex;
        sproxy.receiveMessage(bmsg);
#else
	arr[count].receiveMessage(new (&size, 0) BenchMessage);
#endif
        sendFinishedFlag = 1;	
    }
    
    void receiveMessage(BenchMessage *bmsg){
        
        //CkPrintf("[%d][%d] In Receive Message \n", CkMyPe(), thisIndex);
        
        CkAssert (bmsg->src >= 0 && bmsg->src < nElements);

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

    //All messages received, loadbalance or start next phase
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

    void ResumeFromSync() {
        ComlibResetSectionProxy(&sproxy);
        //myinst.setSourcePe();
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

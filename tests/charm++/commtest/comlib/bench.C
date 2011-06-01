
/***********************
        This benchmark tests all to all personalized communication
   in Charm++ using the communication library.

         To Run
             bench <message_size> <array_elements>

         Defaults
           message size = 128b
           array elements = CkNumPes

         Performance tips
            - Use a maxiter of 1000
            - Use +LBOff to remove loadbalancing overheads

Sameer Kumar 10/28/04      
***********************/

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

#define USELIB  1           // Use comlib or default charm++
#define MAXITER 10          // Number of iterations to run For
                            // benchmark runs this should be atleast a
                            // 1000

#define NUMPASS 1           // Number of all-to-all phases with the
                            // same message size. After NUMPASS phases
                            // the message size will be
                            // increased. Useful while using the
                            // learning framework.

/*readonly*/ CkChareID mid;
/*readonly*/ CProxy_Bench arr;
/*readonly*/ int nElements;

//Old way of creating messages in Charm++
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
	if(m->argc > 2 ) 
            nElements = atoi(m->argv[2]);
        delete m;
        
        //Start the computation
        CkPrintf("Running Bench on %d processors for %d elements with %d byte messages\n", CkNumPes(), nElements, size);
        
        mid = thishandle;        
        //ComlibInstanceHandle tmpInstance = CkGetComlibInstance();
        ComlibInstanceHandle cinst = CkGetComlibInstance();
	
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
        
//        strat->enableLearning();
        cinst.setStrategy(strat);                

        for(count =0; count < nElements; count++)
	  arr[count].insert(cinst);

        arr.doneInserting();

	curTime = CkWallTimer();

        //for(count = 0; count < nElements; count++)            
        //  arr[count].start(size);

        arr.start(size);
    };
    
    void send(void) {
        
      mcount ++;
      
      if (mcount == nElements){
	
	pass ++;
	mcount = 0;
	
	CkPrintf("%d %5.4lf\n", size, (CmiWallTimer() - curTime)*1000/
                 MAXITER);
	curTime = CkWallTimer();
	
	if(pass == NUMPASS)
	  done();
	else {
            //for(int count = 0; count < nElements; count++)            
            //  arr[count].start(size);
            arr.start(size);
        }
      }
    }
    
    void done()
    {	
      superpass ++;
      mcount = 0;
      pass = 0;
      
      if(superpass == 10)
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
          
          //for(int count = 0; count < nElements; count++)            
          //  arr[count].start(size);
          arr.start(size);
      }
    }
};

/******** The all to all benchmar array *************/
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
        //CkPrintf("Migrated to %d\n", CkMyPe());
        //myInst = cinst;
    }
    
    //Send all to all messages
    //proxy arr is the charm++ proxy
    //proxy arrd is the comlib delegated proxy
    void sendMessage()
    {
#ifdef USELIB
        myInst.beginIteration();
#endif        
        for(int count = 0; count < nElements; count ++){
            
            ComlibPrintf("[%d] Sending Message from %d to %d\n", CkMyPe(), thisIndex, count);

#ifdef USELIB
            arrd[count].receiveMessage(new (&msize, 0) BenchMessage); 
#else
	    arr[count].receiveMessage(new (&msize, 0) BenchMessage);
#endif
        }

#ifdef USELIB
        myInst.endIteration();
#endif        
    }

    //receive the all to all messages Once I have everyones message I
    //initiate the next iteration, or call loadbalancing.
    void receiveMessage(BenchMessage *bmsg){
        
        delete bmsg;
        mcount ++;
        
        ComlibPrintf("In Receive Message %d %d %d\n", thisIndex, CkMyPe(), 
                     pass);

        if(mcount == nElements){
            mcount = 0;            
            pass ++;            
            CProxy_Main mainProxy(mid);

            //If I have finished all iterations for this message size,
            //Go back to main and start with a larger message size.
            if(pass == MAXITER){
		pass = 0;                
		mainProxy.send();
            }
            else
                sendMessage();
        }
    }
    
    void start(int messagesize){
        msize = messagesize;

        //Initiate loadbalance at the phases 1 and NUMPASS/2
        //So if NUMPASS = 1, loadbalancing will not be called 
	if(ite % NUMPASS == NUMPASS/2 || ite % NUMPASS == 1) {
            //Call atsync in the middle and in the end
            ComlibPrintf("[%d] Calling Atsync\n", CkMyPe());
            AtSync();
        }
        else
            sendMessage();
        
        //CkPrintf("In Start\n");
        ite ++;
    }

    //Finished loadbalancing
    void ResumeFromSync() {
        //        CkPrintf("%d: resuming\n", CkMyPe());
        myInst.setSourcePe();
	sendMessage();
    }

    //Pack the data for migration
    void pup(PUP::er &p) {
        //if(p.isPacking())
        //  CkPrintf("Migrating from %d\n", CkMyPe());

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


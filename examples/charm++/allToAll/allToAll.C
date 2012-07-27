#include "allToAll.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int numChares;
/*readonly*/ int msgSize;
/*readonly*/ int max_iter;

struct allToAllMsg : public CMessage_allToAllMsg {
  char *data;
};

struct Main : public CBase_Main {
	double start;
	CProxy_allToAll allToAllProxy;
    int         iter;

	Main(CkArgMsg* m) {

        max_iter = 1000;
        numChares = CkNumPes();
        msgSize = 1024;
		// 3D allToAll on a NxNxN array
        if(m->argc >= 2)
        {
            msgSize = atoi(m->argv[1]);
        }
        if(m->argc >= 3)
        {
            numChares = atoi(m->argv[2]);
        }
        
        delete m;

        iter = 0;
		mainProxy = thisProxy;
		// Construct an array of allToAll chares to do the calculation
		allToAllProxy = CProxy_allToAll::ckNew(numChares);
	}

	void allToAllReady() {
		start = CkWallTimer();
		// Broadcast the 'go' signal to the allToAll chare array
		allToAllProxy.doAllToAll();
        
	}

	void nextallToAll() {
		
        iter++;
        if(iter < max_iter)
        {
            allToAllProxy.doAllToAll();
        }else
        {
            double time = CkWallTimer() - start;
            CkPrintf("allToAll on %d cores for msg size: %d per iteration:%f ms\n",
                CkNumPes(), msgSize,  time/max_iter*1000);
            CkExit();
        }
    }

};

struct allToAll : public CBase_allToAll {
	allToAll_SDAG_CODE

	int count;
    int iter;
    allToAllMsg  **msgs;
    int recvCnt;

	allToAll() {
		 __sdag_init();
        iter = 0;
        recvCnt = 0;
		msgs = new allToAllMsg*[numChares];
		for(int i = 0; i < numChares; i++) {
			msgs[i] = new (msgSize) allToAllMsg;
		}

		// reduction to the mainchare to signal that initialization is complete
		contribute(CkCallback(CkReductionTarget(Main,allToAllReady), mainProxy));
	}

	// Sends transpose messages to every other chare
	void sendAllToAll() {
		
        for(int i = thisIndex; i < thisIndex+numChares; i++) {
			int t = i % numChares;
			CkSetRefNum(msgs[t],iter);
			thisProxy[t].getAllToAll(msgs[t]);
		}
	}

	// Sends transpose messages to every other chare
	void getAllToAll() {
	}

    void processAllToAll(allToAllMsg *msg)
    {
        msgs[recvCnt] = msg;
        recvCnt++;
    }


	void finish(){
	    recvCnt = 0;	
        contribute(CkCallback(CkReductionTarget(Main,nextallToAll), mainProxy));
	}

	allToAll(CkMigrateMessage* m) {}
	~allToAll() {}

};

#include "allToAll.def.h"

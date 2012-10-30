#include "allToAll.decl.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int numChares;
/*readonly*/ int msgSize;
/*readonly*/ int max_iter;
/*readonly*/ int msgCount;

struct allToAllMsg : public CMessage_allToAllMsg {
  char *data;
};

struct Main : public CBase_Main {
	double start;
	CProxy_allToAll allToAllProxy;
    int         iter;

	Main(CkArgMsg* m) {

        max_iter = 300;
        numChares = CkNumPes();
        msgSize = 1024;
        msgCount = 1;
        // 3D allToAll on a NxNxN array
        if(m->argc >= 2)
        {
            msgSize = atoi(m->argv[1]);
        }
        if(m->argc >= 3)
        {
            msgCount = atoi(m->argv[2]);
        }
        if(m->argc >= 4)
        {
            max_iter = atoi(m->argv[3]);
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
        if(iter %50 == 0)
            CkPrintf("iter = %d\n", iter);
        if(iter < max_iter)
        {
            allToAllProxy.doAllToAll();
        }else
        {
            double time = CkWallTimer() - start;
            CkPrintf("allToAll on %d cores for msg size: %d\n iteration(ms):%d\t%f\n",
                CkNumPes(), msgSize,  msgSize, time/max_iter*1000);
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
        iter = 0;
        recvCnt = 0;
		msgs = new allToAllMsg*[numChares*msgCount];
		for(int i = 0; i < msgCount*numChares; i++) {
			msgs[i] = new (msgSize) allToAllMsg;
		}

		// reduction to the mainchare to signal that initialization is complete
		contribute(CkCallback(CkReductionTarget(Main,allToAllReady), mainProxy));
	}

	// Sends transpose messages to every other chare
	void sendAllToAll() {
	
        for(int j=0;j<msgCount; j++) {
        for(int i = thisIndex; i < thisIndex+numChares; i++) {
			int t = i % numChares;
			CkSetRefNum(msgs[j*numChares+t],iter);
			thisProxy[t].getAllToAll(msgs[j*numChares+t]);
		}
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

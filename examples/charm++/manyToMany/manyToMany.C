#include "manyToMany.decl.h"
#include <math.h>

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int numChares;
/*readonly*/ int msgSize;
/*readonly*/ int max_iter;

struct manyToManyMsg : public CMessage_manyToManyMsg {
  char *data;
};

struct Main : public CBase_Main {
	double start;
	CProxy_manyToMany manyToManyProxy;
    int         iter;

	Main(CkArgMsg* m) {

        max_iter = 400;
        numChares = sqrt((double)CkNumPes());
        msgSize = 1024;
		// 3D manyToMany on a NxNxN array
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
		// Construct an array of manyToMany chares to do the calculation
		manyToManyProxy = CProxy_manyToMany::ckNew(numChares, numChares);
	}

	void manyToManyReady() {
		start = CkWallTimer();
		// Broadcast the 'go' signal to the manyToMany chare array
		manyToManyProxy.domanyToMany();
        
	}

	void nextmanyToMany() {
		
        iter++;
        if(iter < max_iter)
        {
            manyToManyProxy.domanyToMany();
        }else
        {
            double time = CkWallTimer() - start;
            CkPrintf("manyToMany on %d cores for msg size: %d per iteration:%f ms\n",
                CkNumPes(), msgSize,  time/max_iter*1000);
            CkExit();
        }
    }

};

struct manyToMany : public CBase_manyToMany {
	manyToMany_SDAG_CODE

	int count;
    int iter;
    manyToManyMsg  **msgs;
    int recvCnt;

	manyToMany() {
        iter = 0;
        recvCnt = 0;
		msgs = new manyToManyMsg*[numChares];
		for(int i = 0; i < numChares; i++) {
			msgs[i] = new (msgSize) manyToManyMsg;
		}

		// reduction to the mainchare to signal that initialization is complete
		contribute(CkCallback(CkReductionTarget(Main,manyToManyReady), mainProxy));
	}

	// Sends transpose messages to every other chare
	void sendmanyToMany() {
		
        for(int i = thisIndex.y; i < thisIndex.y+numChares; i++) {
			int t = i % numChares;
			CkSetRefNum(msgs[t],iter);
			thisProxy(thisIndex.x, t).getmanyToMany(msgs[t]);
		}
	}

	// Sends transpose messages to every other chare
	void getmanyToMany() {
	}

    void processmanyToMany(manyToManyMsg *msg)
    {
        msgs[recvCnt] = msg;
        recvCnt++;
    }


	void finish(){
	    recvCnt = 0;	
        contribute(CkCallback(CkReductionTarget(Main,nextmanyToMany), mainProxy));
	}

	manyToMany(CkMigrateMessage* m) {}
	~manyToMany() {}

};

#include "manyToMany.def.h"

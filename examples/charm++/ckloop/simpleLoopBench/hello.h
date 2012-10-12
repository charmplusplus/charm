#ifndef _HELLO_H
#define _HELLO_H

#include "charm++.h"
#include "CkLoopAPI.h"
#include "hello.decl.h"
#include <assert.h>

class Main : public Chare {
private:
	int numElemFinished; //record the number of test instances finished in a timestep
	double timestamp;
	int mainStep; //the global counter of timesteps
	double *mainTimes; //record each timestep from test initiation to test finish (i.e. from the point of main)
    
    int curTestMode; //0: ckLoop; 1: OpenMP

public:
    Main(CkArgMsg* m) ;
    void done(void);
	void exitTest();
    void doTests(CkQdMsg *msg);
    void processCommandLine(int argc,char ** argv);
};

class TestInstance : public CBase_TestInstance {
	int hasTest; //used for reporting statistics
	
    double *allTimes; //record time taken for each timestep
	int *allResults; //record the result of each timestep
	
public:
    TestInstance();
    ~TestInstance() {
		delete [] allTimes;
		delete [] allResults;
	}
    TestInstance(CkMigrateMessage *m) {}
    void doTest(int curstep, int curTestMode);
	void reportSts();
};

class cyclicMap : public CkArrayMap {
public:
    int procNum(int, const CkArrayIndex &idx) {
        int index = *(int *)idx.data();
        int nid = (index/CkMyNodeSize())%CkNumNodes();
        int rid = index%CkMyNodeSize();
        return CkNodeFirst(nid)+rid;
    }
};

#endif

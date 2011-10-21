#ifndef _HELLO_H
#define _HELLO_H

#include "charm++.h"
#include "NodeHelper.h"
#include "hello.decl.h"
#include <assert.h>
//#include "queueing.h"
#include <converse.h>

class Main : public Chare{
private:
    int nodesFinished ;
    int totalElems;
	double wps;
	double calTime; 
	int calibrated;
	int time[4];
	int chunck[4];
	
public:
	 Main(CkArgMsg* m) ;
	void done(void);
	void doTests(CkQdMsg *msg);
	void processCommandLine(int argc,char ** argv);

};
class TestInstance : public CBase_TestInstance {
	double wps;
	int result;
	int flag;
	int helpers;
	double timerec;
	double * allTimes;
public:
    TestInstance();
    ~TestInstance() {}
    TestInstance(CkMigrateMessage *m) {}
    void doTest(int flag,int wps,int chunck, int time);
};

class cyclicMap : public CkArrayMap{
public:
  int procNum(int, const CkArrayIndex &idx){
    int index = *(int *)idx.data();
    int nid = (index/CkMyNodeSize())%CkNumNodes();
    int rid = index%CkMyNodeSize();
    return CkNodeFirst(nid)+rid;
  }
};

#endif

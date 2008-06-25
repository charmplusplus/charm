#include "Main.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CkChareID mainhandle;
/*readonly*/ CkGroupID mCastGrpID;
/*readonly*/ int arrayDimension;
/*readonly*/ int vectorSize;


Main::Main()
{ };

Main::Main(CkMigrateMessage* msg) 
{};

Main::Main(CkArgMsg *m) 
{
	if(m->argc < 3)
	{
		CkPrintf("Please specify the dimension of the array and the size of the vector as the first and the second arguments respectively\n");
		CkExit();
	}
	arrayDimension = atoi(m->argv[1]);
	vectorSize = atoi(m->argv[2]);
	delete m;
	mainProxy = thisProxy;
	mainhandle = thishandle;
	testProxy1D = CProxy_Test1D::ckNew(arrayDimension);
	sectionProxy = new CProxySection_Test1D[arrayDimension];
	CkCallback *cb = new CkCallback(CkIndex_Main::reportSum(NULL), mainProxy);
	testProxy1D.ckSetReductionClient(cb);
	
	//Multicast stuff
	CkArrayID testArrayID = testProxy1D.ckGetArrayID();
	mCastGrpID = CProxy_CkMulticastMgr::ckNew();
	CkMulticastMgr *mCastGrp = CProxy_CkMulticastMgr(mCastGrpID).ckLocalBranch();

	for(int i = 0; i < arrayDimension; i++)
	{
		//creating sectionProxy[i]:
		sectionProxy[i] = CProxySection_Test1D::ckNew(testArrayID, 0, arrayDimension-1, i+1);
		sectionProxy[i].ckSectionDelegate(mCastGrp);
		mCastGrp->setReductionClient(sectionProxy[i], cb);
		//msg for sectionProxy[i]
		DummyMsg *msg = new DummyMsg;
		msg->section = i;
		sectionProxy[i].compute(msg);	//multicast to the section array.
	}
	
	//Quiscence
	int myIndex = CkIndex_Main::QuiDetect();
	CkStartQD(myIndex, &mainhandle);
};

void Main::reportSum(CkReductionMsg *m)
{
	int reducedVecSize = m->getSize() / sizeof(double);
	double *sum = (double*)m->getData();
	CkPrintf("Size of reduced vector is %d.\nContents of reduced vector are:\n", reducedVecSize);
	for(int i = 0; i < reducedVecSize; i++)
	{
		CkPrintf("%f  ",sum[i]);
	}
	CkPrintf("\n");
	delete m;
};

void Main::QuiDetect()
{
	CkPrintf("You should have the following as your results (in any order)\n");
	int temp, factor;
	vector<int> verificationVector(vectorSize);
	for(int i = 0; i < arrayDimension; i++) {
		temp = arrayDimension%(i+1);
		if(temp == 0) {
			factor = arrayDimension/(i+1);
		} else {
			factor = arrayDimension/(i+1) + 1;
		}
		for(int j=0; j < vectorSize; j++) {
			verificationVector[j] = j;
			verificationVector[j] *= factor;
			CkPrintf("%d ", verificationVector[j]);
		}
		CkPrintf("\n");
	}
	/*
	for(int i=0; i < arrayDimension; i++) {
		verificationVector.clear();
		for(int j = 0; j < arrayDimension; j += i) {
			for(int k = 0; k < vectorSize; k++) {
				verificationVector[k] += k;
			}
		}
		for(int p=0; p<vectorSize; p++) {
			CkPrintf("%d ", verificationVector[p]);
		}
		CkPrintf("\n");
	}*/
	CkExit();
};

#include "main.def.h"

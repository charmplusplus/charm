#include "Main.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CkChareID mainhandle;
/*readonly*/ int numElements; 
/*readonly*/ CkGroupID mCastGrpID;
/*readonly*/ int arrayDimensionX;
/*readonly*/ int arrayDimensionY;
/*readonly*/ int vectorSize;

Main::Main()
{ };

Main::Main(CkMigrateMessage* msg) 
{};

Main::Main(CkArgMsg *m) 
{
	if(m->argc < 4)
	{
		CkPrintf("Incorrect usage. Please read the readme.txt file\n");
		CkExit();
	}
	arrayDimensionX = atoi(m->argv[1]);
	arrayDimensionY = atoi(m->argv[2]);
	vectorSize = atoi(m->argv[3]);

	delete m;
	mainProxy = thisProxy;
	mainhandle = thishandle;
	testProxy2D = CProxy_Test2D::ckNew(arrayDimensionX, arrayDimensionY);
	CkCallback *cb = new CkCallback(CkIndex_Main::reportSum(NULL), mainProxy);
	testProxy2D.ckSetReductionClient(cb);
	
	//Multicast stuff
	CkArrayID testArrayID = testProxy2D.ckGetArrayID();
	mCastGrpID = CProxy_CkMulticastMgr::ckNew();
	CkMulticastMgr *mCastGrp = CProxy_CkMulticastMgr(mCastGrpID).ckLocalBranch();
	
	//setting the value of N to be the greater of the two dimensions.
	//(just an arbit decision)
	int N = arrayDimensionX;
	if(arrayDimensionY > arrayDimensionX) {
		N = arrayDimensionY;
	}
	//create N section proxies.
	sectionProxy = new CProxySection_Test2D[N];
	for(int i=0; i < N; i++) {
		/*debug
		CkPrintf("i =%d\n", i);	*/
		//chose which elements from chare array add to the sectionProxy
		sectionProxy[i] = CProxySection_Test2D::ckNew(testArrayID, 0, arrayDimensionX-1, i+1, 0, arrayDimensionY-1, i+1);
		sectionProxy[i].ckSectionDelegate(mCastGrp);
		mCastGrp->setReductionClient(sectionProxy[i], cb);
		//message
		DummyMsg *msg = new DummyMsg;
		msg->section = i;
		sectionProxy[i].compute(msg);
	}
	//Quiscence
	int myIndex = CkIndex_Main::QuiDetect();
	CkStartQD(myIndex, &mainhandle);
};

void Main::reportSum(CkReductionMsg *m)
{
	int reducedVectorSize = m->getSize() / sizeof(double);
	double *sum = (double*)m->getData();
	CkPrintf("reduced vector\n");	//debug
	for(int i = 0; i < reducedVectorSize; i++)
	{
		CkPrintf("%f ", sum[i]);
	}
	CkPrintf("\n");
	delete m;
};

void Main:: QuiDetect()
{
	CkExit();
};
#include "main.def.h"

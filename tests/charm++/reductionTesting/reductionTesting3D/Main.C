#include "Main.h"

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ CkChareID mainhandle;
/*readonly*/ int numElements; 
/*readonly*/ CkGroupID mCastGrpID;
/*readonly*/ int arrayDimensionX;
/*readonly*/ int arrayDimensionY;
/*readonly*/ int arrayDimensionZ;
/*readonly*/ int vectorSize;

Main::Main()
{ };

Main::Main(CkMigrateMessage* msg) 
{};

Main::Main(CkArgMsg *m) 
{
	if(m->argc < 5)
	{
		CkPrintf("Incorrect usage. Please read the readme.txt file\n");
		CkExit();
	}
	arrayDimensionX = atoi(m->argv[1]);
	arrayDimensionY = atoi(m->argv[2]);
	arrayDimensionZ = atoi(m->argv[3]);
	vectorSize = atoi(m->argv[4]);

	delete m;
	mainProxy = thisProxy;
	mainhandle = thishandle;
	testProxy3D = CProxy_Test3D::ckNew(arrayDimensionX, arrayDimensionY, arrayDimensionZ);
	CkCallback *cb = new CkCallback(CkIndex_Main::reportSum(NULL), mainProxy);
	testProxy3D.ckSetReductionClient(cb);
	
	//Multicast stuff
	CkArrayID testArrayID = testProxy3D.ckGetArrayID();
	mCastGrpID = CProxy_CkMulticastMgr::ckNew();
	CkMulticastMgr *mCastGrp = CProxy_CkMulticastMgr(mCastGrpID).ckLocalBranch();
	
	//setting the value of N to be the greater of the two dimensions.
	//(just an arbit decision)
	int N = arrayDimensionX;
	if(arrayDimensionY > arrayDimensionX) {
		N = arrayDimensionY;
	}
	//create N section proxies.
	sectionProxy = new CProxySection_Test3D[N];
	for(int i=0; i < N; i++) {
		//chose elements from chare array to be added to the sectionProxy
		sectionProxy[i] = CProxySection_Test3D::ckNew(testArrayID, 0, arrayDimensionX-1, i+1, 0, arrayDimensionY-1, i+1, 0, arrayDimensionZ-1, i+1);
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

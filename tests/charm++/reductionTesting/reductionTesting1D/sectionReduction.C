#include "sectionReduction.h"

Test1D::Test1D() : doubleVector(vectorSize) 
{
	for(int i = 0; i < vectorSize; i++) {
		doubleVector[i] = i;
	}
};

Test1D::Test1D(CkMigrateMessage *msg)
{ };

void Test1D::compute(DummyMsg *m)
{
	CkSectionInfo cookie;
	cookies.push_back(cookie);
	CkVec<double> myVector(vectorSize);	//creates elements with vectorSize elements
	for(int i = 0; i < vectorSize; i++)	{
		myVector[i] = doubleVector[i];
	}
	int index = m->index;
	CkGetSectionInfo(cookies[index], m);
	CkMulticastMgr *mCastGrp = CProxy_CkMulticastMgr(mCastGrpID).ckLocalBranch();
	CkCallback cb(CkIndex_Main::reportSum(NULL), mainProxy);
	mCastGrp->contribute(sizeof(double)*vectorSize, myVector.getVec(), CkReduction::sum_double, cookies[index], cb);
	delete m;
};

/*Test1D::~Test1D()
{
	delete[] cookies;
};*/
#include "sectionReduction.def.h"

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
	 /* debug
	 * char buf[2048];

	sprintf(buf, "Compute called :: thisIndex = { %d, %d }, m->section = %d", thisIndex.x, thisIndex.y, m->section);	//debugging */

	//cookie stuff.
	int section = m->section;
	int oldLength = cookies.length();
	int newLength = section + 1;
	if(oldLength < newLength) { //only if the section proxy calling 
								//this method is a new section proxy 
								//(that hasn't already called this method 
								//before, increase the size of the cookies array.
		cookies.resize(section+1);
		for (int i = oldLength; i < newLength; i++) {
			CkSectionInfo tempCookie;
			cookies[i] = tempCookie;
		}
	}
	CkGetSectionInfo(cookies[section], m);	//the index of the cookie identifies, which section proxy we are refering to.

    /* DEBUG
	CkSectionInfo si = cookies[section];
	sprintf(buf + strlen(buf), ", Cookie:"
			 "  type = %d"
			 "  pe = %d"
			 "  sCookie = { redNo:%d, val:%p }\n",
			 (int)(si.type), si.pe,
			 si.sInfo.sCookie.redNo,
			 si.sInfo.sCookie.val
			);
    CkPrintf(buf); */
	
	CkVec<double> myVector(vectorSize);	//creates a local vector with vectorSize elements
	for(int i = 0; i < vectorSize; i++)	{
		myVector[i] = doubleVector[i];
	}
	CkMulticastMgr *mCastGrp = CProxy_CkMulticastMgr(mCastGrpID).ckLocalBranch();
	CkCallback cb(CkIndex_Main::reportSum(NULL), mainProxy);
	mCastGrp->contribute(sizeof(double)*vectorSize, myVector.getVec(), CkReduction::sum_double, cookies[section], cb);
	delete m;
};

/*Test1D::~Test1D()
{
	delete[] cookies;
};*/
#include "sectionReduction.def.h"

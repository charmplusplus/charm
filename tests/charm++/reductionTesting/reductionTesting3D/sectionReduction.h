#ifndef __SECTIONREDUCTION_H__
#define __SECTIONREDUCTION_H__
#include <cklists.h>	//for CkVector
#include <iostream>
#include "sectionReduction.decl.h"
#include "main.decl.h"
#include "ckmulticast.h"

using namespace std;

extern CkGroupID mCastGrpID;
extern CProxy_Main mainProxy;
extern int arrayDimensionX;
extern int vectorSize;
extern int arrayDimensionY;

class DummyMsg: public CkMcastBaseMsg, public CMessage_DummyMsg
{
	public:
		int section;
};

class Test3D: public CBase_Test3D
{
	private:
        CkSectionInfo testCookie;
		CkVec<CkSectionInfo> cookies;
		CkVec<double> doubleVector;

	public:
		Test3D();
		Test3D(CkMigrateMessage *msg);
		void compute(DummyMsg *m);
		//virtual ~Test3D();
};

#endif

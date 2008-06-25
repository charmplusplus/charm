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
extern int arrayDimension;
extern int vectorSize;

class DummyMsg: public CkMcastBaseMsg, public CMessage_DummyMsg
{
	public:
		int section;
};

class Test1D: public CBase_Test1D
{
	private:
		CkVec<CkSectionInfo> cookies;
		CkVec<double> doubleVector;

	public:
		Test1D();
		Test1D(CkMigrateMessage *msg);
		void compute(DummyMsg *m);
		//virtual ~Test1D();
};

#endif

#ifndef __MAIN_H__
#define __MAIN_H__
#include <iostream>
#include <vector>
#include "main.decl.h"
#include "sectionReduction.h"
//#include "ckmulticast.h"

using namespace std;

class Main: public CBase_Main
{
	private:
		CProxy_Test1D testProxy1D;
		CProxySection_Test1D *sectionProxy;

	public:
		Main();
		Main(CkArgMsg *m);
		Main(CkMigrateMessage* msg);
		void QuiDetect();
		void reportSum(CkReductionMsg *m);
};

#endif

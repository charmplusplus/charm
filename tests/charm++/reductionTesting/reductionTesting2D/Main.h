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
		CProxy_Test2D testProxy2D;
		CProxySection_Test2D *sectionProxy;

	public:
		Main();
		Main(CkArgMsg *m);
		Main(CkMigrateMessage* msg);
		void reportSum(CkReductionMsg *m);
		void QuiDetect();
};

#endif

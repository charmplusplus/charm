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
		CProxy_Test3D testProxy3D;
		CProxySection_Test3D *sectionProxy;

	public:
		Main();
		Main(CkArgMsg *m);
		Main(CkMigrateMessage* msg);
		void reportSum(CkReductionMsg *m);
		void QuiDetect();
};

#endif

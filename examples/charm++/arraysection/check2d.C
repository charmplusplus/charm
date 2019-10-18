#include <math.h>
#include "check2d.decl.h"
#include "ckmulticast.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Check checkArray;
/* readonly */ int numchares_x;
/* readonly */ int numchares_y;

struct sectionBcastMsg : public CkMcastBaseMsg, public CMessage_sectionBcastMsg {
	int k;
	sectionBcastMsg(int _k) : k(_k) {}
	void pup(PUP::er &p) {
		CMessage_sectionBcastMsg::pup(p);
		p|k;
	}
};

class Main : public CBase_Main {
	int sum;

public:
	Main(CkArgMsg* msg) {
		if (msg->argc < 3) {
			ckout << "Usage: " << msg->argv[0] << " [number of x_chares] [number of y_chares]" << endl;
			CkExit(1);
		}
		numchares_x = atoi(msg->argv[1]);
		numchares_y = atoi(msg->argv[2]);
		ckout << "Numchares_x: " << numchares_x << " Numchares_y: " << numchares_y << endl;
		sum = 0;
		mainProxy = thisProxy;
		checkArray = CProxy_Check::ckNew(numchares_x, numchares_y);
		checkArray.createSection();
	}

	void done1It(int q, int output[]) {
		CkAssert(q == 2);
		int expected[2];
		int n_x = ((numchares_x - 1) % 2 == 0) ? (numchares_x - 1) : (numchares_x - 2);
		int n_y = ((numchares_y - 1) % 2 == 0) ? (numchares_y - 1) : (numchares_y - 2);
		int sumofIndices_x = n_x * (n_x + 2) / 4;  // even numbered chares are part of the section
		int sumofIndices_y = n_y * (n_y + 2) / 4;
		expected[0] = sumofIndices_x * ceil(numchares_y / 2.0);
		expected[1] = sumofIndices_y * ceil(numchares_x / 2.0);
		CkAssert((output[0] - expected[0]) == 0);
		CkAssert((output[1] - expected[1]) == 0);
		ckout << "Vector reduction: " << output[0] << " " << output[1] << endl;
		ckout << "Test passed successfully" << endl;
		CkExit();
	}
};

class Check : public CBase_Check {
	CProxySection_Check secProxy;

public:
	Check() {}

	void createSection() {
		if (thisIndex.x == 0 && thisIndex.y == 0) {
			// Use index range to create a section [lbound:ubound:stride]
			secProxy = CProxySection_Check::ckNew(checkArray.ckGetArrayID(), 0, numchares_x - 1, 2, 0, numchares_y - 1, 2);
			sectionBcastMsg *msg = new sectionBcastMsg(0);
			secProxy.recvMsg(msg);
		}
	}

	void recvMsg(sectionBcastMsg *msg) {
		ckout << "ArrayIndex: [" << thisIndex.x << ", " << thisIndex.y << "] - " << CkMyPe() << endl;
		int k = msg->k;
		std::vector<int> outVals(2);
		outVals[0] = k + thisIndex.x;
		outVals[1] = k + thisIndex.y;
		CkSectionInfo cookie;
		CkGetSectionInfo(cookie, msg);
		CkCallback cb(CkReductionTarget(Main, done1It), mainProxy);
		CProxySection_Check::contribute(outVals, CkReduction::sum_int, cookie, cb);
		delete msg;
	}
};

#include "check2d.def.h"

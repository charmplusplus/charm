#include <math.h>
#include "check.decl.h"
#include "ckmulticast.h"

/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Check checkArray;
/* readonly */ int numchares;

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
		if (msg->argc < 2) {
			ckout << "Usage: " << msg->argv[0] << " [number of chares]" << endl;
			CkExit(1);
		}
		numchares = atoi(msg->argv[1]);
		ckout << "Numchares: " << numchares << endl;
		sum = 0;
		mainProxy = thisProxy;
		checkArray = CProxy_Check::ckNew(numchares);
		checkArray.createSection();
	}

	void done(int q, int output[]) {
		CkAssert(q == 2);
		int expected[2];
		int n = ((numchares - 1) % 2 == 0) ? (numchares - 1) : (numchares - 2);
		int sumofIndices = n * (n + 2) / 4;  // even numbered chares are part of the section
		expected[0] = 1 * ceil(numchares / 2.0) + sumofIndices;
		expected[1] = 2 * ceil(numchares / 2.0) + sumofIndices;
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
		if (thisIndex == 0) {
			std::vector<CkArrayIndex> elems;  // add array indices
			elems.reserve(numchares / 2);
			for (int i = 0; i < numchares; i += 2) {
				elems.emplace_back(i);
			}
			secProxy = CProxySection_Check::ckNew(checkArray.ckGetArrayID(), elems);
			sectionBcastMsg *msg = new sectionBcastMsg(1);
			secProxy.recvMsg(msg);
		}
	}

	void recvMsg(sectionBcastMsg *msg) {
		ckout << "ArrayIndex: " << thisIndex << " - " << CkMyPe() << endl;
		int k = msg->k;
		std::vector<int> outVals(2);
		outVals[0] = k + thisIndex;
		outVals[1] = k + 1 + thisIndex;
		CkSectionInfo cookie;
		CkGetSectionInfo(cookie, msg);
		CkCallback cb(CkReductionTarget(Main, done), mainProxy);
		CProxySection_Check::contribute(outVals, CkReduction::sum_int, cookie, cb);
		delete msg;
	}
};

#include "check.def.h"

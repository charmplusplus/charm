#include "check.decl.h"
/* readonly */ CProxy_Main mainProxy;
/* readonly */ CProxy_Check checkArray;
/* readonly */ int numchares;

class Main : public CBase_Main {
	int sum;
	public:
	Main(CkArgMsg* msg){
		numchares = atoi(msg->argv[1]);
		ckout<<" numchares: "<<numchares<<endl;
		checkArray = CProxy_Check::ckNew(numchares);
		checkArray.scattervData();
		sum = 0;
		mainProxy = thisProxy;
		CkCallback cb(CkIndex_Main::done(), thisProxy);
		CkStartQD(cb);
	}
	//Main(CkMigrateMessage* msg){}
	//void pup(PUP::er &p){}
	void done(){
		ckout<<"Main::Done "<<endl;
		CkExit();
	}
};


class Check : public CBase_Check {
	public:
	Check() {}
	Check(CkMigrateMessage* msg) {}
	void scattervData(){
		if(thisIndex == 0){
			int *arr = new int[2*numchares];
			for(int i=0; i<2*numchares; i++)
				arr[i] = i;
			CkVec<CkArrayIndex1D> elems;    // add array indices
			CkVec<int> disp, cnt;           // byte offset
			int ndest = 0;
			for (int i=0; i<numchares; i+=2, ndest++){
				elems.push_back(CkArrayIndex1D(i));
				disp.push_back(i*sizeof(int));
				cnt.push_back(2*sizeof(int));
			}
			CkScattervWrapper w(arr, ndest, disp.getVec(), elems.getVec(), cnt.getVec());
			CkPrintf("[%d]In scattervData : calling recvScatterv\n", thisIndex);
			int dummysize;
			checkArray.recvScatterv(w, dummysize, 1.0);
			free(arr);
		}
	}

	void recvScatterv(int *arr, int size, double pos){
		CkPrintf("[%d] In recvScatterv, buflen: %d (ints), pos: %lf \n", thisIndex, size, pos);
		for(int i=0; i<size; i++){
			//CkPrintf("[%d] arr[%d]: %d \n", thisIndex, i, arr[i]);
			CkAssert(arr[i] == thisIndex+i);
        }
	}

};


#include "check.def.h"

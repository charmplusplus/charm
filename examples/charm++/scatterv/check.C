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
		checkArray.scatterData();
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
	//CProxySection_Check secProxy;
	public:
	Check() {}
	Check(CkMigrateMessage* msg) {}		
	void scatterData(){
                int *arr = new int[2*numchares];
		for(int i=0; i<2*numchares; i++)
			arr[i] = i;
		if(thisIndex == 0){
			CkVec<CkArrayIndex1D> elems;    // add array indices
 			CkVec<int> disp, cnt;           // byte offset
			int ndest = 0;
			for (int i=0; i<numchares; i+=2, ndest++){
				elems.push_back(CkArrayIndex1D(i));
				disp.push_back(i*sizeof(int));
				cnt.push_back(2*sizeof(int));
                        }
                        CkScatterWrapper w(arr, ndest, disp.getVec(), elems.getVec(), cnt.getVec());
			CkPrintf("In createsection : calling recvScatter\n");
 			checkArray.recvScatter(w, 10, 1.0);
		}
	}

	void recvScatter(int *arr, int size, double pos){
		CkPrintf("[%d] In recvScatter arr: %p, size: %d, pos: %lf \n", thisIndex, arr, size, pos);
		int numints = size/sizeof(int);
		for(int i=0; i<numints; i++)
			CkPrintf("[%d] arr[%d]: %d \n", thisIndex, i, arr[i]);
	}

};


#include "check.def.h"

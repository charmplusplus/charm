/***********************************
*	Author : Amit Sharma
*	Date : 12/04/2004
************************************/

#ifndef _AGENT_H
#define _AGENT_H

#include "CentralLB.h"
#include "topology.h"

class Agent {
public:
	int npes;
	
	typedef struct _Elem {
		int pe;
		double Cost;
		_Elem(): pe(-1), Cost(-1.0) {}
	}Elem;
	
	Elem *preferred_list;

//public:
	Agent(int p): npes(p) { }
	virtual ~Agent() { }

	virtual Elem* my_preferred_procs(int *existing_map,int object,int *trialpes,int metric){ return NULL; }
};

class TopologyAgent : public Agent {
//protected:
public:
	CentralLB::LDStats* stats;
	LBTopology *topo;
	
	int **commObjs;
	int **hopCount;
//public:			
	TopologyAgent(CentralLB::LDStats* lbDB,int p);
	~TopologyAgent() { }
	Agent::Elem* my_preferred_procs(int *existing_map,int object,int *trialpes,int metric);
	static int compare(const void *p,const void *q);
};

class comlibAgent : public Agent {
	protected:
		void* comlibstats;
	public:
		comlibAgent(int p): Agent(p) { }
		~comlibAgent() { }

	Agent::Elem* my_preferred_procs(int *existing_map,int object,int *trialpes,int metric);

};

class MulticastAgent : public Agent {
protected:
	struct MInfo {
  	  int nbytes;
	  int messages;
          CkVec<int>  objs;
          MInfo(): nbytes(0), messages(0) {}
          MInfo(int b, int n): nbytes(b), messages(n) {}
	};
	CkVec<MInfo> mcastList;
        CkVec<int> *objmap;	    // list of mcast involved for every object
        int  nobj;
public:
	MulticastAgent(BaseLB::LDStats* lbDB, int p);
	virtual ~MulticastAgent() { delete [] objmap; }

	virtual Elem* my_preferred_procs(int *existing_map,int object,int *trialpes,int metric);
	
};

#endif

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
	
	typedef struct{
		int pe;
		int Cost;
	}Elem;
	
	Elem *preferred_list;

//public:
	Agent(int p): npes(p) { }
	~Agent() { }

	virtual Elem* my_preferred_procs(int *existing_map,int object,int *trialpes){ }
};

class topoAgent : public Agent {
//protected:
public:
	CentralLB::LDStats* stats;
	//Elem *preferred_list;
	LBTopology *topo;
	
	/*typedef struct{
		int obj;
		int bytesComm;
	}CommObj;
*/
	//CommObj
	int **commObjs;
	int **hopCount;
//public:			
	topoAgent(CentralLB::LDStats* lbDB,void* comlibDB,LBTopology *topology,int p): stats(lbDB), topo(topology), Agent(p){
		stats->makeCommHash();
		preferred_list = new Elem[p];
		commObjs = new int*[stats->n_objs];
		for(int i=0;i<stats->n_objs;i++){
			commObjs[i] = new int[stats->n_objs];
			for(int j=0;j<stats->n_objs;j++)
				commObjs[i][j] = 0;
		}

		hopCount = new int*[npes];
		for(int i=0;i<npes;i++){
			hopCount[i] = new int[npes];
			for(int j=0;j<npes;j++)
				hopCount[i][j] = 0;
		}

		for(int i=0;i<stats->n_comm;i++){
			//DO I consider other comm too....i.e. to or from a processor
			//CkPrintf("in loop..\n");
			LDCommData &cdata = stats->commData[i];
			if(cdata.from_proc() || cdata.receiver.get_type() != LD_OBJ_MSG)
    		continue;
    	int senderID = stats->getHash(cdata.sender);
			CmiAssert(senderID < stats->n_objs);
    	int recverID = stats->getHash(cdata.receiver.get_destObj());
    	CmiAssert(recverID < stats->n_objs);
		
			//Check with Gengbin if 2 different commData have the same pair of processors as role reversed
			//for(int j=0;j<num_comm;j++)
				
			commObjs[senderID][recverID] += cdata.bytes;
			commObjs[recverID][senderID] += cdata.bytes;
		}
		
		//Pre-calculate the hop counts
		/*for(int i=0;i<npes;i++){
			CkPrintf("for proc num :%d calculating hop counts\n",i);
			for(int j=0;j<npes;j++)
				if(i!=j && !hopCount[i][j]){
					hopCount[i][j]=topo->get_hop_count(i,j);
					hopCount[j][i]=hopCount[i][j];
				}
		}*/	
	}
	~topoAgent() { }
	Agent::Elem* my_preferred_procs(int *existing_map,int object,int *trialpes);
	static int compare(const void *p,const void *q);
};

class comlibAgent : public Agent {
	protected:
		void* comlibstats;
	public:
		comlibAgent(int p): Agent(p) { }
		~comlibAgent() { }

	Agent::Elem* my_preferred_procs(int *existing_map,int object,int *trialpes);

};

#endif

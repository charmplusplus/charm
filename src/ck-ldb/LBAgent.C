/***********************************
*	Author : Amit Sharma
*	Date : 12/04/2004
************************************/


#include <math.h>

//#include "charm.h"

//#include "topology.h"
#include "LBAgent.h"


int topoAgent :: compare(const void *p,const void *q){
	return (((Elem*)p)->Cost - ((Elem*)q)->Cost);
}
		

Agent::Elem* topoAgent :: my_preferred_procs(int *existing_map,int object,int *trialpes) {

	int *procs_list;
	
	//int *preferred_list;

	for(int i=0;i<npes;i++){
		preferred_list[i].pe = -1;
		preferred_list[i].Cost = 0;
	}

	if(trialpes == NULL)
		*procs_list = -1;
	else
		procs_list = trialpes;

	//Before everything...construct a list of comm of this object with all the other objects

	int proc=0;
	int index=0;
	int cost=0;
	int preflistSize = 0;

	if(procs_list[0]!=-1)
		proc = procs_list[index];
	
	/*int j=0;
	while(procs_list[j]!=-1){
		CkPrintf(" %d,%d ",j,procs_list[j]);
		j++;
	}*/
		
	//CkPrintf("in preferred....before loop\n");
	
	while(1){
		
		cost=0;
		if(!stats->procs[proc].available){
			//CkPrintf("some problem\n");
			if(procs_list[0]==-1){
				proc++;
				if(proc == npes)
					break;
			}
			else {
				index++;
				if(procs_list[index] == -1)
					break;
				proc = procs_list[index];
			}
			continue;
		}
		
		//CkPrintf("before cost cal..\n");
		for(int i=0;i<stats->n_objs;i++){
			if(existing_map[i]!=-1 && commObjs[object][i]!=0){
				//CkPrintf("before calling get hop count..proc:%d,existing map:%d\n",proc,existing_map[comm[i].obj]);
				if(hopCount[proc][existing_map[i]])
					cost += hopCount[proc][existing_map[i]]*commObjs[object][i];
				else{
					hopCount[proc][existing_map[i]]=topo->get_hop_count(proc,existing_map[i]);
					hopCount[existing_map[i]][proc]=hopCount[proc][existing_map[i]];
					cost += hopCount[proc][existing_map[i]]*commObjs[object][i];
				}
				//CkPrintf("after calling get hop count...\n");
			}
		}
		//CkPrintf("after cost cal..\n");
		preferred_list[preflistSize].pe = proc;
		preferred_list[preflistSize].Cost = cost;
		preflistSize++;
		
		if(procs_list[0]==-1){
			proc++;
			//CkPrintf("at wrong place\n");
			if(proc == npes)
				break;
		}
		else {
			index++;
			//CkPrintf("right place ,index:%d",index);
			if(procs_list[index] == -1)
				break;
			proc = procs_list[index];
		}

	}

	//Sort all the elements of the preferred list in increasing order
	qsort(preferred_list,preflistSize,sizeof(Elem),&compare);
	//for(int k=0;k<preflistSize;k++)
		//CkPrintf("pe:%d cost:%d\n",preferred_list[k].pe,preferred_list[k].Cost);
	
	return preferred_list;
}

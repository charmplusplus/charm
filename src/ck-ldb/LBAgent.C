/***********************************
*	Author : Amit Sharma
*	Date : 12/04/2004
************************************/


#include <math.h>
#include <string.h>

#include "LBAgent.h"

#define PREF_RET_SIZE 5

TopologyAgent::TopologyAgent(CentralLB::LDStats* lbDB,int p): stats(lbDB), Agent(p){
	int i;

	LBtopoFn topofn;
	topofn = LBTopoLookup(_lbtopo);
  if (topofn == NULL) {
  	char str[1024];
   	CmiPrintf("LBAgent> Fatal error: Unknown topology: %s. Choose from:\n", _lbtopo);
   	printoutTopo();
   	sprintf(str, "LBAgent> Fatal error: Unknown topology: %s", _lbtopo);
   	CmiAbort(str);
  }
	topo = topofn(p);

	stats->makeCommHash();
	preferred_list = new Elem[p];
	commObjs = new int*[stats->n_objs];
	for(i=0;i<stats->n_objs;i++){
		commObjs[i] = new int[stats->n_objs];
		for(int j=0;j<stats->n_objs;j++)
			commObjs[i][j] = 0;
	}

	hopCount = new int*[npes];
	for(i=0;i<npes;i++){
		hopCount[i] = new int[npes];
		for(int j=0;j<npes;j++)
			hopCount[i][j] = 0;
	}
	//commObjs = comm;

	for(i=0;i<stats->n_comm;i++){
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
}

int TopologyAgent :: compare(const void *p,const void *q){
	return (int)(((Elem*)p)->Cost - ((Elem*)q)->Cost);
}
		

Agent::Elem* TopologyAgent :: my_preferred_procs(int *existing_map,int object,int *trialpes,int metric) {

	int tp;   // tmp var
	int *procs_list;
	
	//int *preferred_list;
	//CkPrintf("npes:%d\n",npes);
	for(int i=0;i<npes;i++){
		preferred_list[i].pe = -1;
		preferred_list[i].Cost = 0;
	}

	int preflistSize=0;
	
	if(metric==1){ //First metric
		//CkPrintf("in first metric\n");
		if(trialpes == NULL) {
			procs_list = &tp;
			*procs_list = -1;
		}
		else
			procs_list = trialpes;

		//Before everything...construct a list of comm of this object with all the other objects
		int proc=0;
		int index=0;
		int cost=0;

		if(procs_list[0]!=-1)
			proc = procs_list[index];
	
		while(1){
			cost=0;
			if(!stats->procs[proc].available){
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
		
			//First metric --- hops*bytes
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
				}
			}
			preferred_list[preflistSize].pe = proc;
			preferred_list[preflistSize].Cost = cost;
			preflistSize++;
		
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
		}
	}
	
	if(metric==2){
		//Second metric --- place the object closer to maximum comm. processor
		int i=0;
		int max_neighbors = topo->max_neighbors();
		int *comProcs = new int[npes];
		for(i=0;i<npes;i++)
			comProcs[i]=0;
		int max_comm=0;
		int max_proc=-1;
		int *neigh;
		for(i=0;i<stats->n_objs;i++){
			if(existing_map[i]!=-1 && commObjs[object][i]!=0){
				comProcs[existing_map[i]] += commObjs[object][i];
				if(comProcs[existing_map[i]] > max_comm){
					max_comm = comProcs[existing_map[i]];
					max_proc = existing_map[i];
				}
			}
		}
		int num_neigh=0;
		int done=0,j=0;
		if(max_proc!=-1){
			i=0;
			while(trialpes[i]!=-1){
				if(max_proc==trialpes[i]){
					preferred_list[0].pe = max_proc;
					preferred_list[0].Cost = max_comm;
					preflistSize++;
					done = 1;
				}
			}
			if(!done){
				neigh = new int[max_neighbors];
				topo->neighbors(max_proc,neigh,num_neigh);
				while(trialpes[i]!=-1){
					for(j=0;j<num_neigh;j++)
						if(trialpes[i]==neigh[j]){
							preferred_list[preflistSize].pe = neigh[j];
							preferred_list[preflistSize].Cost = comProcs[neigh[j]];
							preflistSize++;
							done=1;
						}
				}
			}
			if(!done){
				int *secondneigh = new int[max_neighbors];
				int k=0;
				i=0;
				int num=num_neigh;
				for(k=0;k<num;k++){
					topo->neighbors(neigh[k],secondneigh,num_neigh);
					while(trialpes[i]!=-1){
						for(j=0;j<num_neigh;j++)
							if(trialpes[i]==secondneigh[j]){
								preferred_list[preflistSize].pe = secondneigh[j];
								preferred_list[preflistSize].Cost = comProcs[secondneigh[j]];
								preflistSize++;
								done=1;
							}
					}
				}
			}

		}
	}
		/***************************************************************************/

	//Third metric -- as sugggested by Sanjay

	//Sort all the elements of the preferred list in increasing order
	Agent::Elem *prefreturnList = new Elem[PREF_RET_SIZE+1];
	int *taken_proc = new int[preflistSize];
	double min_cost=preferred_list[0].Cost;
	int min_cost_index=0;

	//prefreturnList[0].pe=preferred_list[min_cost_index].pe;
	//prefreturnList[0].Cost=preferred_list[min_cost_index].Cost;
	
	int s=0;
	int flag=0;
	int u=0;
	
	for(s=0;s<preflistSize;s++)
		taken_proc[s]=0;
	for(s=0;s<PREF_RET_SIZE;s++){
		for(u=0;u<preflistSize;u++)
			if(!taken_proc[u]){
				min_cost=preferred_list[u].Cost;
				min_cost_index=u;
				break;
			}
		if(u==preflistSize)
			break;
		for(int t=u;t<preflistSize;t++){
			if(preferred_list[t].Cost <= min_cost && !taken_proc[t]){
				min_cost = preferred_list[t].Cost;
				min_cost_index=t;
				flag=1;
			}
		}
		if(flag){
			taken_proc[min_cost_index]=1;
			prefreturnList[s].pe=preferred_list[min_cost_index].pe;
			prefreturnList[s].Cost=preferred_list[min_cost_index].Cost;
			flag=0;
		}
	}
	prefreturnList[s].pe=-1;
	prefreturnList[s].Cost=-1;
	//qsort(preferred_list,preflistSize,sizeof(Elem),&compare);
	//for(int k=0;k<preflistSize;k++)
		//CkPrintf("pe:%d cost:%d\n",preferred_list[k].pe,preferred_list[k].Cost);
	
	return prefreturnList;
	//return preferred_list;
}

/*****************************************************************************
		Multicast Agent 
*****************************************************************************/

MulticastAgent::MulticastAgent(BaseLB::LDStats* stats, int p): Agent(p)
{
  stats->makeCommHash();
  // build multicast knowledge
  nobj = stats->n_objs;
  objmap = new CkVec<int> [nobj];
  for (int com = 0; com < stats->n_comm;com++) {
    LDCommData &commData = stats->commData[com];
    if (commData.recv_type()!=LD_OBJLIST_MSG) continue;
      // create a multicast instance
    mcastList.push_back(MInfo(commData.bytes, commData.messages));
    int mID = mcastList.size()-1;
    MInfo &minfo = mcastList[mID];
    int sender = stats->getHash(commData.sender);
      // stores all multicast that this object (sender) participated
    objmap[sender].push_back(mID);
      // stores all objects that belong to this multicast
    minfo.objs.push_back(sender);
    int nobjs;
    LDObjKey *objs = commData.receiver.get_destObjs(nobjs);
    for (int i=0; i<nobjs; i++) {
       int receiver = stats->getHash(objs[i]);
       if((sender == -1)||(receiver == -1)) {
         if (_lb_args.migObjOnly()) continue;
         else CkAbort("Error in search\n");
       }
       objmap[receiver].push_back(mID);
       minfo.objs.push_back(receiver);
    }
  }
}

Agent::Elem* MulticastAgent::my_preferred_procs(int *existing_map,int object,int *trialpes,int metric){ 
  int i;
  // check all multicast this object participated
  CmiAssert(object < nobj);

  double * comcosts = new double [npes];
  memset(comcosts, 0, sizeof(double)*npes);
  double alpha = _lb_args.alpha();
  double beta = _lb_args.beta();

    // all multicast this object belongs to
  CkVec<int> &mlist = objmap[object];
  // traverse all multicast participated
  // find out which processor it communicates the most
  for (i=0; i<mlist.size(); i++) {
     MInfo &minfo = mcastList[mlist[i]];
     for (int obj=0; obj<minfo.objs.size(); obj++) {
       int pe = existing_map[obj];
       if (pe == -1) continue;		// not assigned yet
       comcosts[pe] += minfo.messages * alpha + minfo.nbytes * beta;
     }
  }
  // find number of processors with non-0 cost
  int count = 0;
  for (i=0; i<npes; i++) {
    if (comcosts[i] != 0.0) count++;
  }
  Elem *prefered = new Elem[count+1];
  for (i=0; i<count; i++) {
    // find the maximum
    Elem maxp;	  // cost default -1
    for (int j=0; j<npes; j++)
      if (comcosts[j] != 0.0 && comcosts[j] > maxp.Cost) {
        maxp.pe = j;
        maxp.Cost = comcosts[j];
      }
    CmiAssert(maxp.pe!=-1);
    prefered[i] = maxp;
    comcosts[maxp.pe] = 0.0;
  }

  delete [] comcosts;
  return prefered;
}





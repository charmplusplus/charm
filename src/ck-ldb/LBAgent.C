/***********************************
*	Author : Amit Sharma
*	Date : 12/04/2004
************************************/


#include <math.h>
#include <string.h>

//#include "charm.h"

//#include "topology.h"
#include "LBAgent.h"


int topoAgent :: compare(const void *p,const void *q){
	return (int)(((Elem*)p)->Cost - ((Elem*)q)->Cost);
}
		

Agent::Elem* topoAgent :: my_preferred_procs(int *existing_map,int object,int *trialpes) {
	int tp;   // tmp var
	int *procs_list;
	
	//int *preferred_list;

	for(int i=0;i<npes;i++){
		preferred_list[i].pe = -1;
		preferred_list[i].Cost = 0;
	}

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
    mcastList.push_back(MInfo(commData.bytes, commData.messages));
    int mID = mcastList.size()-1;
    MInfo &minfo = mcastList[mID];
    int sender = stats->getHash(commData.sender);
    objmap[sender].push_back(mID);
    minfo.objs.push_back(sender);
    int nobjs;
    LDObjKey *objs = commData.receiver.get_destObjs(nobjs);
    for (int i=0; i<nobjs; i++) {
       int receiver = stats->getHash(objs[i]);
       if((sender == -1)||(receiver == -1))
         if (_lb_args.migObjOnly()) continue;
         else CkAbort("Error in search\n");
       objmap[receiver].push_back(mID);
       minfo.objs.push_back(receiver);
    }
  }
}

Agent::Elem* MulticastAgent::my_preferred_procs(int *existing_map,int object,int *trialpes){ 
  int i;
  // check all multicast this object participated
  CmiAssert(object < nobj);

  CkVec<int> &mlist = objmap[object];
  double * comcosts = new double [npes];
  memset(comcosts, 0, sizeof(double)*npes);
  double alpha = _lb_args.alpha();
  double beeta = _lb_args.beeta();

  // traverse all multicast participated
  // find out which processor it communicates the most
  for (i=0; i<mlist.size(); i++) {
     MInfo &minfo = mcastList[mlist[i]];
     for (int obj=0; obj<minfo.objs.size(); obj++) {
       int pe = existing_map[obj];
       if (pe == -1) continue;		// not assigned
       comcosts[pe] += minfo.messages * alpha + minfo.nbytes * beeta;
     }
  }
  // find number of non-0 cost processors
  int count = 0;
  for (i=0; i<npes; i++) {
    if (comcosts[i] != 0.0) count++;
  }
  Elem *prefered = new Elem[count+1];
  for (i=0; i<count; i++) {
    // find the maximum
    Elem maxp;
    for (int j=0; j<npes; j++)
      if (prefered[j].Cost > maxp.Cost) {
        maxp = prefered[j];
      }
    prefered[i] = maxp;
  }

  delete [] comcosts;
  return prefered;
}





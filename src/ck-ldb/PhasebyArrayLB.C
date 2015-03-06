/**
 * \addtogroup CkLdb
*/
/*@{*/

#include "PhasebyArrayLB.h"

extern LBAllocFn getLBAllocFn(const char *lbname);

CreateLBFunc_Def(PhasebyArrayLB, "Load balancer which balances many arrays together, specifically for CPAIMD")

#include "PhasebyArrayLB.def.h"

PhasebyArrayLB::PhasebyArrayLB(const CkLBOptions &opt): CBase_PhasebyArrayLB(opt)
{
  lbname = (char*)"PhasebyArrayLB";
  if (CkMyPe() == 0)
    CkPrintf("[%d] PhasebyArrayLB created\n",CkMyPe());

  const char *lbs = theLbdb->loadbalancer(opt.getSeqNo());
  
  char *lbcopy = strdup(lbs);
  char *p = strchr(lbcopy, ':');
  if (p==NULL) return;
  p++;
  LBAllocFn fn = getLBAllocFn(p);
  if (fn == NULL) {
  	CkPrintf("LB> Invalid load balancer: %s.\n", p);
  	CmiAbort("");
  }
  lb = (CentralLB *)fn();
}

bool PhasebyArrayLB::QueryBalanceNow(int _step)
{
  return true;
}

void PhasebyArrayLB::copyStats(BaseLB::LDStats *stats,BaseLB::LDStats *tempStats){
	int i;
	tempStats->nprocs() = stats->nprocs();
	tempStats->n_objs = stats->n_objs;
	tempStats->n_comm = stats->n_comm;
	tempStats->n_migrateobjs = stats->n_migrateobjs;
	tempStats->hashSize = stats->hashSize;
	if(tempStats->hashSize && stats->objHash!=NULL){
		tempStats->objHash = new int[tempStats->hashSize];
		for(int i=0;i<tempStats->hashSize;i++)
			tempStats->objHash[i]=stats->objHash[i];
	}
	else
		tempStats->objHash=NULL;
	tempStats->objData.resize(tempStats->n_objs);
	tempStats->from_proc.resize(tempStats->n_objs);
	tempStats->to_proc.resize(tempStats->n_objs);
	tempStats->commData.resize(tempStats->n_comm);
	for(i=0;i<tempStats->n_objs;i++){
		tempStats->objData[i]=stats->objData[i];
		tempStats->from_proc[i]=stats->from_proc[i];
		tempStats->to_proc[i]=stats->to_proc[i];
	}
	for(i=0;i<tempStats->n_comm;i++)
		tempStats->commData[i]=stats->commData[i];
	
	tempStats->procs = new BaseLB::ProcStats[tempStats->nprocs()];
	for(i=0; i<tempStats->nprocs(); i++)
		tempStats->procs[i]=stats->procs[i];
}

void PhasebyArrayLB::updateStats(BaseLB::LDStats *stats,BaseLB::LDStats *tempStats){
	tempStats->hashSize = stats->hashSize;
	if(tempStats->hashSize && stats->objHash!=NULL){
		tempStats->objHash = new int[tempStats->hashSize];
		for(int i=0;i<tempStats->hashSize;i++)
			tempStats->objHash[i]=stats->objHash[i];
	}
	else
		tempStats->objHash=NULL;

	for(int i=0;i<tempStats->n_objs;i++)
		tempStats->objData[i]=stats->objData[i];
	
}

void PhasebyArrayLB::work(LDStats *stats){
	//It is assumed that statically placed arrays are set non-migratable in the application
	tempStats = new BaseLB::LDStats;

	copyStats(stats,tempStats);
	int obj, i;
	int flag=0;
	LDObjData *odata;
	
	
	odata = &(tempStats->objData[0]);
	omids.push_back(odata->omID());
	if(odata->migratable)
		migratableOMs.push_back(true);
	else
		migratableOMs.push_back(false);

	for(i=0;i<tempStats->n_objs; i++){
		odata = &(tempStats->objData[i]);
		for(int j=0;j<omids.size();j++)
			if(odata->omID()==omids[j]){
				flag=1;
				break;
			}
		
		if(flag==1){
			flag=0;
		}
		else{
			omids.push_back(odata->omID());
			if(odata->migratable)
				migratableOMs.push_back(true);
			else
				migratableOMs.push_back(false);
		}
	}
	
	for(i=0;i<omids.size();i++){
		//copy to_proc from previous iteration to from_proc for this iteration
		LDOMid  omid = omids[i];
		//Set other objects as background load
		if(migratableOMs[i]){
			for (obj = 0; obj < tempStats->n_objs; obj++) {
		  	odata = &(tempStats->objData[obj]);
		  	if (odata->omID() != omid)
					odata->migratable=false;
 			}
			//Call a strategy here
			lb->work(tempStats);
			if(i!=omids.size()-1){
				for(obj = 0; obj < tempStats->n_objs; obj++)
					tempStats->from_proc[obj]=tempStats->to_proc[obj];
				updateStats(stats,tempStats);
			}
		}
	}
	//Copy to stats array
	for(obj = 0; obj < tempStats->n_objs; obj++)
		stats->to_proc[obj]=tempStats->to_proc[obj];
	tempStats->clear();
	omids.free();
	migratableOMs.free();
}

/*@}*/

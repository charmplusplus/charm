/**
   @addtogroup ComlibCharmStrategy
   @{
   @file 
*/

#include "EachToManyMulticastStrategy.h"
#include "string.h"

#include "AAPLearner.h"
#include "AAMLearner.h"


//Group Constructor
EachToManyMulticastStrategy::EachToManyMulticastStrategy(int substrategy,
		CkGroupID src,
		CkGroupID dest,
		int n_srcpes, 
		int *src_pelist,
		int n_destpes, 
		int *dest_pelist) 
: RouterStrategy(substrategy), CharmStrategy() {
	ComlibPrintf("[%d] EachToManyMulticast group constructor\n",CkMyPe());

	setType(GROUP_STRATEGY);

	//CkGroupID gid;
	//gid.setZero();
	ginfo.setSourceGroup(src, src_pelist, n_srcpes);    
	ginfo.setDestinationGroup(dest, dest_pelist, n_destpes);

	//Written in this funny way to be symmetric with the array case.
	//ginfo.getDestinationGroup(gid, destpelist, ndestpes);
	//ginfo.getCombinedPeList(pelist, npes);

	commonInit(ginfo.getCombinedCountList()); // The array returned by getCombinedCountList is deleted inside commonInit
}

//Array Constructor
EachToManyMulticastStrategy::EachToManyMulticastStrategy(int substrategy, 
		CkArrayID src, 
		CkArrayID dest, 
		int nsrc, 
		CkArrayIndex *srcelements, 
		int ndest, 
		CkArrayIndex *destelements)
: RouterStrategy(substrategy), CharmStrategy() {
	ComlibPrintf("[%d] EachToManyMulticast array constructor. nsrc=%d ndest=%d\n",CkMyPe(), nsrc, ndest);

	setType(ARRAY_STRATEGY);
	ainfo.setSourceArray(src, srcelements, nsrc);
	ainfo.setDestinationArray(dest, destelements, ndest);

	int *count = ainfo.getCombinedCountList();
	//ainfo.getSourcePeList(nsrcPe, srcPe);
	//ainfo.getDestinationPeList(ndestPe, destPe);

	commonInit(count);
}

extern char *routerName;
//Common initialization for both group and array constructors
void EachToManyMulticastStrategy::commonInit(int *count) {

	setBracketed();
	//setForwardOnMigration(1);

	if(CkMyPe() == 0 && routerName != NULL){
		if(strcmp(routerName, "USE_MESH") == 0)
			routerIDsaved = USE_MESH;
		else if(strcmp(routerName, "USE_GRID") == 0)
			routerIDsaved = USE_GRID;
		else  if(strcmp(routerName, "USE_HYPERCUBE") == 0)
			routerIDsaved = USE_HYPERCUBE;
		else  if(strcmp(routerName, "USE_DIRECT") == 0)
			routerIDsaved = USE_DIRECT;        
		else  if(strcmp(routerName, "USE_PREFIX") == 0)
			routerIDsaved = USE_PREFIX;        

		//Just for the first step. After learning the learned
		//strategies will be chosen
		//router = NULL;
	}

	//ComlibPrintf("Creating Strategy %d\n", routerID);

	// finish the creation of the RouterStrategy superclass
	bracketedUpdatePeKnowledge(count);
	delete [] count;
	//rstrat = NULL;
}

EachToManyMulticastStrategy::~EachToManyMulticastStrategy() {
}


void EachToManyMulticastStrategy::insertMessage(CharmMessageHolder *cmsg){

  //  cmsg -> checkme();

	ComlibPrintf("[%d] EachToManyMulticast: insertMessage\n", CkMyPe());

	envelope *env = UsrToEnv(cmsg->getCharmMessage());

	if(cmsg->dest_proc == IS_BROADCAST) {
		//All to all multicast

		// not necessary to set cmsg since the superclass will take care
		//cmsg->npes = ndestPes;
		//cmsg->pelist = destPes;

		//Use Multicast Learner (Foobar will not work for combinations
		//of personalized and multicast messages

		// this handler will ensure that handleMessage is called if direcly
		// sent, or deliver is called if normal routing path is taken
		CmiSetHandler(env, CkpvAccess(comlib_handler));

		//Collect Multicast Statistics
		RECORD_SENDM_STATS(getInstance(), env->getTotalsize(), 
				cmsg->pelist, cmsg->npes);
	}
	else {
		//All to all personalized

		//Collect Statistics
		RECORD_SEND_STATS(getInstance(), env->getTotalsize(), 
				cmsg->dest_proc);
	}

	RouterStrategy::insertMessage(cmsg);
}


void EachToManyMulticastStrategy::pup(PUP::er &p){

	ComlibPrintf("[%d] EachToManyMulticastStrategy::pup called for %s\n", CkMyPe(), 
			p.isPacking()?"packing":(p.isUnpacking()?"unpacking":"sizing"));

	RouterStrategy::pup(p);
	CharmStrategy::pup(p);

}



void EachToManyMulticastStrategy::deliver(char *msg, int size) {
	ComlibPrintf("[%d] EachToManyMulticastStrategy::deliver for %s\n",
			CkMyPe(), isAllToAll()?"multicast":"personalized");
	
	envelope *env = (envelope *)msg;
	RECORD_RECV_STATS(myHandle, env->getTotalsize(), env->getSrcPe());

	if (isAllToAll()){
		ComlibPrintf("Delivering via localMulticast()\n");
		localMulticast(msg);
	}
	else { 
		if (getType() == GROUP_STRATEGY) {
			ComlibPrintf("Delivering via personalized CkSendMsgBranchInline\n");
			CkUnpackMessage(&env);
			CkSendMsgBranchInline(env->getEpIdx(), EnvToUsr(env), CkMyPe(), env->getGroupNum());
		}
		else if (getType() == ARRAY_STRATEGY) {
			//    	  ComlibPrintf("[%d] Delivering via ComlibArrayInfo::deliver()\n", CkMyPe());
			//      ComlibArrayInfo::deliver(env);

			ComlibPrintf("[%d] Delivering via ComlibArrayInfo::localBroadcast()\n", CkMyPe());
			ainfo.localBroadcast(env);

		}
	}
}

void EachToManyMulticastStrategy::localMulticast(void *msg){
	register envelope *env = (envelope *)msg;
	CkUnpackMessage(&env);
	ComlibPrintf("localMulticast calls ainfo.localBroadcast()\n");
	ainfo.localBroadcast(env);
}

void EachToManyMulticastStrategy::notifyDone() {
	if (!getOnFinish().isInvalid()) getOnFinish().send(0);
	RouterStrategy::notifyDone();
}

/*@}*/

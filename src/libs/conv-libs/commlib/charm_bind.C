/***************************************************
 * File : charm_bind.C
 *
 * Author: Krishnan V
 *
 * Charm++ binding. Uses charm++ messages
 ****************************************************/
#include "charm++.h"
#include "commlib.h"

#include "../../../ck.h"
#include "../../../envelope.h"
#include "../../../trace.h"

class envelope;

#if 0
//void CComlibEachToManyMulticast(id, ep, msg, bocnum, npe, pelist, ref)
void CComlibEachToManyMulticast(comID id, int ep, void *msg, int bocnum, int npe, int *pelist)
{
  if (msg == NULL) {
  	//EachToManyMulticastWithRef(id, 0, (void *)msg, npe, pelist, ref);
  	EachToManyMulticast(id, 0, (void *)msg, npe, pelist);
	return;
  }
  	
  ENVELOPE *env;
  int len, queueing, priobits; unsigned int *prioptr;
  CldInfoFn ifn = (CldInfoFn)CmiHandlerToFunction(CpvAccess(CkInfo_Index));
  CldPackFn pfn;
 
  env = ENVELOPE_UPTR(msg);

  int type=BroadcastBocMsg;
  SetEnv_msgType(env, type);
  SetEnv_boc_num(env, bocnum);
  SetEnv_EP(env, ep);
 
  ifn((void *)env, &pfn, &len, &queueing, &priobits, &prioptr);
  if (pfn) {
      pfn(&env);
      ifn((void *)env, &pfn, &len, &queueing, &priobits, &prioptr);
  }
  CmiSetHandler(env,CpvAccess(HANDLE_INCOMING_MSG_Index));
  //EachToManyMulticastWithRef(id, len, (void *)env, npe, pelist, ref);
  EachToManyMulticast(id, len, (void *)env, npe, pelist);
  if((type!=QdBocMsg)&&(type!=QdBroadcastBocMsg)&&(type!=LdbMsg))
    QDCountThisCreation(npe);
}
#endif

void CComlibEachToManyMulticast(comID id, int ep, void *msg, int bocnum, int npe, int *pelist)
{
  int len, queueing, priobits; 
  unsigned int *prioptr;
  CldPackFn pfn;

  if (msg == NULL) {
  	EachToManyMulticast(id, 0, (void *)msg, npe, pelist);
	return;
  }

  register envelope *env = UsrToEnv(msg);
  _CHECK_USED(env);
  env->setMsgtype(ForBocMsg);
  env->setEpIdx(ep);
  env->setGroupNum(bocnum);
  env->setSrcPe(CkMyPe());

//  _infoFn(msg, &pfn, &len, &queueing, &priobits, &prioptr);
  _packFn((void **)&env);
  _infoFn(env, &pfn, &len, &queueing, &priobits, &prioptr);

  CmiSetHandler(env, _charmHandlerIdx);
  _SET_USED(env, 1);

/*
  for (int i=0; i<npe-1; i++)
     CmiSyncSend(pelist[i], len, env);
  CmiSyncSendAndFree(pelist[npe-1], len, env);
*/

  _TRACE_CREATION_N(env, npe);

  CpvAccess(_qd)->create();
  EachToManyMulticast(id, len, (void *)env, npe, pelist);

  _STATS_RECORD_SEND_BRANCH_N(npe);
     
//CkPrintf("EachToManyMulticast: len:%d npe:%d done\n", len, npe);
}



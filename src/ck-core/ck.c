#include "charm.h"

#include "trace.h"
#include "converse.h"

#include <varargs.h>

void CkUnpack(ENVELOPE **msg);
void CkPack(ENVELOPE **msg);
void CkInfo(void *msg, CldPackFn *pfn, int *len, int *qs, int *pb, unsigned int **pp);

extern void *FIFO_Create();
extern CHARE_BLOCK *CreateChareBlock();

CpvStaticDeclare(int, num_exits);
CpvStaticDeclare(int, num_endcharms);
CpvDeclare(int, CkInfo_Index);

void ckModuleInit()
{
   CpvInitialize(int, num_exits);
   CpvInitialize(int, num_endcharms);
   CpvInitialize(int, CkInfo_Index);

   CpvAccess(num_exits)=0;
   CpvAccess(num_endcharms)=0;
   CpvAccess(CkInfo_Index) = CldRegisterInfoFn(CkInfo);
}


/*************************************************************************
  EXIT PROTOCOL FOR CHARM

  How do the CkExit and CkEndCharm protocols work?

  CkExits send a message with an iteration number whose value is num_exits
  CkEndCharms send -1 in the message ..

  To prevent the sending of two messages for a particular CkExit,
    *  the seq number in the message should equal the value of num_exits for
       a CkExit to be processed (thus no two CkExits with the same seq number
       are processed).
  Also, CkEndCharm has a synchronization : the broadcast for statistics
  collection is done only after all processors have reported a CkEndCharm.

  The scheme is necessitated by the possibility of use of more than one 
  "DoCharm()" (now Scheduler()) and CkExits in each of them.

  The control flow - Any processor calling CkExit or CkEndCharm sends a 
  BroadcastExitMessage to node 0 which broadcasts to ExitMessage on all 
  processors.

  ExitMessage handles CkExit and CkEndCharm requests differently.

**************************************************************************/


CkExit()
{
	int *msg;

	msg = (int *) CkAllocMsg(sizeof(int));
	CkMemError(msg);
	*msg = CpvAccess(num_exits);

	GeneralSendMsgBranch(CsvAccess(CkEp_Stat_BroadcastExitMessage),
			msg, 0, BocMsg, StatisticBocNum);
}


CkEndCharm()
{
	int *msg;

	msg = (int *) CkAllocMsg(sizeof(int));
	CkMemError(msg);
	*msg = -1;

	GeneralSendMsgBranch(CsvAccess(CkEp_Stat_BroadcastExitMessage),
			msg, 0, BocMsg, StatisticBocNum);
}

BroadcastExitMessage(usr, data)
void *usr, *data;
{
/* This function is executed only on node 0 - corresponds to 
                 CsvAccess(CkEp_Stat_BroadcastExitMessage) */

	int *msg;
	
	if (*((int *)usr) == -1) { /* For CkEndCharm */
		CpvAccess(num_endcharms)++ ;
        	if( CpvAccess(num_endcharms) < CmiNumPes() ) 
	        	return;
	}
	else {  /* For CkExit */
		if(*((int *)usr) < CpvAccess(num_exits))
                	 return;
		CpvAccess(num_exits)++;
	}
	
	msg = (int *) CkAllocMsg(sizeof(int));
	CkMemError(msg);
	*msg = *((int *)usr);
	GeneralBroadcastMsgBranch(CsvAccess(CkEp_Stat_ExitMessage), msg, 
			BroadcastBocMsg, StatisticBocNum);
	CpvAccess(disable_sys_msgs) = 1;
}

ExitMessage(usr, data)
void *usr, *data;
{
        if(*((int *)usr) == -1) /* If the user called CkEndCharm */
	{
		SendNodeStatistics();
		send_log();
		if ( CmiMyPe() != 0 && CpvAccess(CtrRecdTraceMsg) 
		 		    && CpvAccess(RecdStatMsg) ) 
			ExitNode();
	}
	else /* If the user called CkExit */
	{
	        CkEndCharm();
		if(CmiMyPe())
			CpvAccess(num_exits)++;
	}
}


SendNodeStatistics()
{
	(*(CsvAccess(EpInfoTable)[CsvAccess(CkEp_Stat_Data)].function)) 
								(NULL,NULL);
}


ExitNode()
{
	char *msg;
	ENVELOPE *env;

	/* close_log(); moved to convcore.c */
	if (CmiMyPe() == 0)
	{
		/* First print out statistics. */
		PrintOutStatistics();
	}

	/* Complete the loop */	
        CsdExitScheduler();
}






/**********************************************************************
 * These are utility routines for chares 
***********************************************************************/

ChareExit()
{
	SetID_chare_magic_number(CpvAccess(currentChareBlock)->selfID,-1);
	CmiFree(CpvAccess(currentChareBlock));
}


CHARE_BLOCK *CreateChareBlock(sizeData, kind, magic)
int sizeData, magic, kind;
{
  CHARE_BLOCK *p = (CHARE_BLOCK *)CmiAlloc(sizeof(CHARE_BLOCK) + sizeData);
  CkMemError(p);
  SetID_chare_magic_number(p->selfID, magic);
  SetID_onPE(p->selfID, CmiMyPe());
  SetID_chareBlockPtr(p->selfID, p);
  p->charekind = kind;

  /* the chare data area is just after the chare-block, by default */
  p->chareptr = (void *)(p+1); 
  return((void *) p);
}

IsChareLocal(chareid)
ChareIDType * chareid;
{
	if (GetID_onPE((*chareid)) == CmiMyPe()) return 1;
	return 0;
}

void *GetChareDataPtr(chareid)
ChareIDType * chareid;
{
	return ((CHARE_BLOCK *) GetID_chareBlockPtr((*chareid))) + 1;
}

MyChareID(pChareID)
ChareIDType * pChareID;
{
    *pChareID = CpvAccess(currentChareBlock)->selfID;
}


/* Deleted already commented out MyParentID function : SANJEEV May 24, 93 */

MainChareID(pChareID)
ChareIDType * pChareID;
{
	SetID_onPE((*pChareID), 0);
	if (CmiMyPe() == 0)
		SetID_chare_magic_number((*pChareID),
		    GetID_chare_magic_number(CpvAccess(mainChareBlock)->selfID));
	else
		SetID_chare_magic_number((*pChareID), CpvAccess(mainChare_magic_number));
	SetID_chareBlockPtr((*pChareID), CpvAccess(mainChareBlock));
}



/* this is the general CreateChare call: all the user level CreateChare
   calls are mapped to this call: they include 

 	CreateChare(Charename, Entry, Msg, [vid [,destPE]]) 

   If vid is NULL_VID it is a CreateChare call ("without ID"). 
   if DestPe is CK_PE_ANY  then it may go to any destination node
   if DestPe is not CK_PE_SPECIAL then the message is bound for a regular destination

*/

CreateChare(id, Entry, Msg, vid, destPE)
int id;
EntryNumType Entry;
void *Msg;
ChareIDType *vid;
int destPE;
{
  ENVELOPE *env;
  CHARE_BLOCK *vidblock;

  if (id!=CsvAccess(EpInfoTable)[Entry].chareindex) 
    CmiPrintf("** ERROR ** Illegal combination of CHAREINDEX/EP in CreateChare\n");

  TRACE(CmiPrintf("[%d] CreateChare: Entry=%d\n", CmiMyPe(), Entry));
  
  CpvAccess(nodecharesCreated)++;
  env = ENVELOPE_UPTR(Msg);
  
  SetEnv_EP(env, Entry);
  
  if (vid != NULL_VID) {
    vidblock = (CHARE_BLOCK *)CreateChareBlock(0, CHAREKIND_UVID, rand());
    vidblock->x.vid_queue = (void *)FIFO_Create();
    (*vid) = vidblock->selfID;
    SetEnv_vidPE(env, GetID_onPE(vidblock->selfID));
    SetEnv_vidBlockPtr(env, GetID_chareBlockPtr(vidblock->selfID));
  } else {
    SetEnv_vidPE(env, -1);
    SetEnv_vidBlockPtr(env, NULL);
  }
  
  if(CpvAccess(traceOn))
    trace_creation(NewChareMsg, Entry, env);
  QDCountThisCreation(1);

  SetEnv_msgType(env, NewChareMsg);
  if (destPE == CK_PE_ANY) destPE = CLD_ANYWHERE;
  CmiSetHandler(env, CpvAccess(HANDLE_INCOMING_MSG_Index));
  CldEnqueue(destPE, env, CpvAccess(CkInfo_Index));
}


SendMsg(Entry, Msg, pChareID)
int Entry;
void * Msg;
ChareIDType * pChareID;
{
  ENVELOPE * env;
  int destPE = GetID_onPE((*pChareID));

  CpvAccess(nodeforCharesCreated)++;
  env = ENVELOPE_UPTR(Msg);
  SetEnv_msgType(env, ForChareMsg);
  SetEnv_EP(env, Entry);
  SetEnv_chareBlockPtr(env, GetID_chareBlockPtr((*pChareID)));
  SetEnv_chare_magic_number(env, GetID_chare_magic_number((*pChareID)));
  QDCountThisCreation(1);
  if(CpvAccess(traceOn))
    trace_creation(GetEnv_msgType(env), Entry, env);
  CmiSetHandler(env, CpvAccess(HANDLE_INCOMING_MSG_Index));
  CldEnqueue(destPE, env, CpvAccess(CkInfo_Index));
}

/*****************************************************************/
/** Gets reference number.					**/
/*****************************************************************/
GetRefNumber(msg)
void *msg;
{
	ENVELOPE *env = (ENVELOPE *) ENVELOPE_UPTR(msg);

	return GetEnv_ref(env);
}


/*****************************************************************/
/** Sets reference number.					**/
/*****************************************************************/
SetRefNumber(msg, number)
void *msg;
int number;
{
	ENVELOPE *env = (ENVELOPE *) ENVELOPE_UPTR(msg);

	SetEnv_ref(env, number);
}


void CkSetQueueing(usrptr, kind)
void *usrptr;
int kind;
{
  SetEnv_queueing(ENVELOPE_UPTR(usrptr), kind);
}

/*****************************************************************************
 * Load balancer needs.
 *****************************************************************************/

void CkUnpack(ENVELOPE **henv)
{
  ENVELOPE *envelope = *henv;
  void *unpackedUsrMsg; 
  void *usrMsg = USER_MSG_PTR(envelope); 
  if(CpvAccess(traceOn)) 
    trace_begin_unpack(); 
  (*(CsvAccess(MsgToStructTable)[GetEnv_packid(envelope)].unpackfn)) 
    (usrMsg, &unpackedUsrMsg); 
  if(CpvAccess(traceOn)) 
    trace_end_unpack(); 
  if (usrMsg != unpackedUsrMsg) 
    /* else unpacked in place */ 
  { 
    int temp_i; 
    int temp_size; 
    char *temp1, *temp2; 
    /* copy envelope */ 
    temp1 = (char *) envelope; 
    temp2 = (char *) ENVELOPE_UPTR(unpackedUsrMsg); 
    temp_size = (char *) usrMsg - temp1; 
    for (temp_i = 0; temp_i<temp_size; temp_i++) 
    *temp2++ = *temp1++; 
    CmiFree(envelope); 
    envelope = ENVELOPE_UPTR(unpackedUsrMsg); 
   } 
   SetEnv_isPACKED(envelope, UNPACKED); 
  *henv = envelope;
}

void CkPack(ENVELOPE **henv)
{
  ENVELOPE *env = *henv;
  if (GetEnv_isPACKED(env) == UNPACKED) {
    /* needs packing and not already packed */ 
    int size; 
    char *usermsg, *packedmsg; 
    /* make it +ve to connote a packed msg */ 
    SetEnv_isPACKED(env, PACKED); 
    usermsg = USER_MSG_PTR(env); 
    if(CpvAccess(traceOn)) 
      trace_begin_pack(); 
    (*(CsvAccess(MsgToStructTable)[GetEnv_packid(env)].packfn)) 
      (usermsg, &packedmsg, &size); 
    if(CpvAccess(traceOn)) 
      trace_end_pack(); 
    if (usermsg != packedmsg) { 
      /* Free the usermsg here. */ 
      CkFreeMsg(usermsg); 
      env = ENVELOPE_UPTR(packedmsg); 
    }
  }
  *henv = env;
}


void CkInfo(void *msg, CldPackFn *pfn, int *len,
	    int *queueing, int *priobits, unsigned int **prioptr)
{
  ENVELOPE *env = (ENVELOPE *)msg;
  *pfn = (CldPackFn)CkPack;
  *len = GetEnv_TotalSize(env);
  *queueing = GetEnv_queueing(env);
  *priobits = GetEnv_priosize(env);
  *prioptr = GetEnv_priobgn(env);
}


CkEnqueue(env)
void *env;
{
  CsdEnqueueGeneral(env,
    GetEnv_queueing(env),
    GetEnv_priosize(env),
    GetEnv_priobgn(env));
}

/************************************************************************
 *
 * CkPrioConcat
 *
 * Copies all the priority bits from the bitvector in 'srcmsg' onto
 * the bitvector in 'dstmsg', then, if there is any space left in the
 * bitvector of 'dstmsg', that space is filled by bits taken from the
 * lsb of 'delta'.
 *
 * The code works as follows:
 *
 * step 1: Copy old bitvector onto new. Always copies a multiple of
 * 32 bits, therefore, may copy some "padding" bits.  The number of
 * padding bits copied can be found in 'padbits'.
 *
 * step 2: move bits in delta to msb-end.
 *
 * step 3: if any padding-bits were copied, overwrite them with a
 * piece of delta.
 *
 * step 4: if padding-bits were insufficient to hold all of delta,
 * store remainder of delta in next word.
 *
 ************************************************************************/

#define CINTBITS (sizeof(int)*8)

void CkPrioConcatFn(srcmsg, dstmsg, delta)
void *srcmsg;
void *dstmsg;
unsigned int delta;
{
  int padbits, deltabits;
  ENVELOPE *srcenv = ENVELOPE_UPTR(srcmsg);
  ENVELOPE *dstenv = ENVELOPE_UPTR(dstmsg);
  int srcbits = GetEnv_priosize(srcenv);
  int dstbits = GetEnv_priosize(dstenv);
  int srcwords = (srcbits+CINTBITS-1)/CINTBITS;
  int dstwords = (dstbits+CINTBITS-1)/CINTBITS;
  unsigned int *srcptr = GetEnv_prioend(srcenv) - srcwords;
  unsigned int *dstptr = GetEnv_prioend(dstenv) - dstwords;
  deltabits = dstbits - srcbits;
  if (deltabits < 0) {
    CmiPrintf("CkPrioConcat: prio-bits from source message don't fit in destination message.\n");
    exit(1);
  }
  if (deltabits > CINTBITS) {
    CmiPrintf("CkPrioConcat: prio-bits from source message plus bits of delta don't fill destination-message.\n");
    exit(1);
  }
  while (srcbits>0) { *dstptr++ = *srcptr++; srcbits -= CINTBITS; }
  padbits = -srcbits;
  delta <<= (CINTBITS-deltabits);
  if (padbits) {
    dstptr[-1] &= (((unsigned int)(-1))<<padbits);
    dstptr[-1] |= (delta>>(CINTBITS-padbits));
  }
  if (deltabits>padbits) dstptr[0] = (delta<<padbits);
}

int CkPrioSizeBitsFn(msg) void *msg;
{
    return GetEnv_priosize(ENVELOPE_UPTR(msg));
}

int CkPrioSizeBytesFn(msg) void *msg;
{
    return GetEnv_priobytes(ENVELOPE_UPTR(msg));
}

int CkPrioSizeWordsFn(msg) void *msg;
{
    return GetEnv_priowords(ENVELOPE_UPTR(msg));
}

unsigned int *CkPrioPtrFn(msg) void *msg;
{
    return GetEnv_priobgn(ENVELOPE_UPTR(msg));
}

/***************************************************************************
 * RCS INFORMATION:
 *
 *	$RCSfile$
 *	$Author$	$Locker$		$State$
 *	$Revision$	$Date$
 *
 ***************************************************************************
 * DESCRIPTION:
 *
 ***************************************************************************
 * REVISION HISTORY:
 *
 * $Log$
 * Revision 2.1  1998-03-02 14:58:04  jyelon
 * Forgot to check these in last time.
 *
 *
 ***************************************************************************/

#ifndef CHARM_H
#define CHARM_H

#include "converse.h"

/******************************************************************************
 *
 * Converse Concepts, renamed to CK
 *
 *****************************************************************************/

#define CK_QUEUEING_FIFO   CQS_QUEUEING_FIFO
#define CK_QUEUEING_LIFO   CQS_QUEUEING_LIFO
#define CK_QUEUEING_IFIFO  CQS_QUEUEING_IFIFO
#define CK_QUEUEING_ILIFO  CQS_QUEUEING_ILIFO
#define CK_QUEUEING_BFIFO  CQS_QUEUEING_BFIFO
#define CK_QUEUEING_BLIFO  CQS_QUEUEING_BLIFO

#define CkTimer()  	((int)(CmiTimer() * 1000.0))
#define CkUTimer()	((int)(CmiWallTimer() * 1000000.0))
#define CkHTimer()	((int)(CmiWallTimer() / 3600.0))

#define CTimer()  	((int)(CmiTimer() * 1000.0))
#define CUTimer()	((int)(CmiWallTimer() * 1000000.0))
#define CHTimer()	((int)(CmiWallTimer() / 3600.0))

/******************************************************************************
 *
 * Unclassified / Miscellaneous
 *
 *****************************************************************************/

#ifdef DEBUG
#define TRACE(p) p
#else
#define TRACE(p)
#endif

#ifndef NULL
#define NULL 0
#endif

#define CHARRED(x) ((char *) (x))

#define CINTBITS (sizeof(int)*8)


/******************************************************************************
 *
 * Miscellaneous Constants
 *
 *****************************************************************************/

#define NO_PACK		0
#define UNPACKED	1
#define PACKED		2

#define CHARE 		53
#define BOC 		35

#define CHARM 		0
#define CHARMPLUSPLUS 	1

#define ACCUMULATOR	0
#define MONOTONIC	1
#define TABLE 		2

#define NULL_VID       NULL
#define NULL_PE        NULL
#define NULL_PACK_ID    0  

#define CK_PE_SPECIAL(x) ((x)>=0xFFF0)
#define CK_PE_ALL        (0xFFF0)
#define CK_PE_ALL_BUT_ME (0xFFF1)
#define CK_PE_ANY        (0xFFF2)
#define CK_PE_INVALID    (0xFFF3)

#define QDBocNum         1  /* Quiescecence Detection */
#define WOVBocNum        2  /* write once variables   */
#define TblBocNum        3  /* dynamic table boc      */
#define DynamicBocNum    4  /* to manage dynamic boc  */
#define StatisticBocNum  5  /* to manage statistics   */
#define NumSysBoc        6  /* a sentinel for this list */

#define MaxBocs		100
#define PSEUDO_Max	20

#define MainInitEp            0
#define NumHostSysEps         0
#define NumNodeSysEps         0

#define CHAREKIND_CHARE    0   /* Plain old chare */
#define CHAREKIND_BOCNODE  1   /* BOC node */
#define CHAREKIND_UVID     2   /* Unfilled-VID */
#define CHAREKIND_FVID     3   /* Filled-VID */

#define _CK_VARSIZE_UNIT 8

/******************************************************************************
 *
 * Unclassified / Miscellaneous
 *
 *****************************************************************************/

typedef int 		PeNumType;
typedef int 		EntryPointType;
typedef int 		EntryNumType;
typedef int 		ChareNumType;
typedef int 		ChareNameType;
typedef int 		MsgTypes;
typedef int 		MsgCategories;
typedef int 		START_LOOP;
typedef void 		FUNC();
typedef int 		(*FUNCTION_PTR)();
typedef int 		FunctionRefType;
typedef int 		WriteOnceID;    

/******************************************************************************
 *
 * Message Categories and Types
 *
 *****************************************************************************/

#define	IMMEDIATEcat	0
#define USERcat    	1

/* USERcat */
#define NewChareMsg  		0
#define NewChareNoBalanceMsg    1
#define ForChareMsg  		2
#define BocInitMsg   		3
#define BocMsg       		4
#define TerminateToZero 	5   /* never used??? */
#define TerminateSys		6   /* never used??? */
#define InitCountMsg 		7
#define ReadVarMsg   		8
#define ReadMsgMsg 		9
#define BroadcastBocMsg 	10
#define DynamicBocInitMsg 	11

/* IMMEDIATEcat */
#define LdbMsg			12  /* never used??? */
#define VidSendOverMsg          13
#define QdBocMsg		14
#define QdBroadcastBocMsg	15
#define ImmBocMsg               16
#define ImmBroadcastBocMsg      17
#define InitBarrierPhase1       18
#define InitBarrierPhase2       19

/******************************************************************************
 *
 * Data Structures.
 *
 *****************************************************************************/

typedef struct chare_id_type  {
  unsigned short        onPE;
  unsigned short        magic;
  struct chare_block   *chareBlockPtr;
} ChareIDType;

typedef struct chare_block { 
  char charekind;                   /* CHAREKIND: CHARE BOCNODE UVID FVID */
  ChareIDType selfID;               /* My chare ID. */
  union {
    ChareNumType boc_num;         /* if a BOC node */
    ChareIDType  realID;          /* if a Filled-VID */
    struct fifo_queue *vid_queue; /* if an Unfilled-VID */
  } x;
  void *chareptr ;		   /* Pointer to the data area of the chare */
  double data[1];                   /* Pad it to 8 bytes */
} CHARE_BLOCK ;  

typedef struct ep_struct {
  char *name;
  FUNCTION_PTR function;
  int language;
  int messageindex;
  int chareindex;
  int chare_or_boc;
  int threaded;
} EP_STRUCT;

typedef struct msg_struct {
  int size;
  FUNCTION_PTR packfn;
  FUNCTION_PTR unpackfn;
  FUNCTION_PTR alloc;
} MSG_STRUCT;

typedef struct mono_struct {
  FUNCTION_PTR updatefn;
} MONO_STRUCT;

typedef struct acc_struct {
  FUNCTION_PTR addfn;
  FUNCTION_PTR combinefn;
} ACC_STRUCT;
 
typedef struct {
  int id;
  int Penum;
  char *dataptr;
  int AlreadyDone;
  struct chare_id_type CID;
  int NumChildren;
  EntryPointType EP;
} ACC_DATA;

typedef struct {
  int id;
  int time;
  char *dataptr;
  int ismodified;
} MONO_DATA;

typedef struct table_struct {
  FUNCTION_PTR hashfn;
} TABLE_STRUCT;
 
typedef struct pseudo_struct {
  int type;
  int language ;
  char *name;
  FUNCTION_PTR initfn;
  union {
    MONO_STRUCT mono;
    ACC_STRUCT acc;
    TABLE_STRUCT tbl;
  } pseudo_type;
} PSEUDO_STRUCT;

typedef struct bocinit_queue {
  void **block;
  short block_len;
  short first;
  short length;
} BOCINIT_QUEUE;

typedef struct DUMMY_MSG {
  int x;
} DUMMY_MSG;

/******************************************************************************
 *
 * The Charm Envelope
 *
 * Current envelope size: 256 bits = 32 bytes = 4 doubles.
 *
 * Note: the user-data area is aligned to a 64-bit boundary.  Therefore,
 * there is no point to trimming the envelope unless you can save 64 bits.
 *
 * save 32 bits: remove 'event'.  Easy with ifdefs, doubles SUPER_INSTALL time.
 * save 16 bits: remove 'pe'.     Easy with ifdefs, doubles SUPER_INSTALL time.
 * save 16 bits: change TotalSize to a magnitude.  Inefficient.
 * save 16 bits: could eliminate priosize, by moving it into priority. Clumsy.
 * save  8 bits: remove msgType by replacing HANDLE_X_MSG.  Hard.
 * save 14 bits: turn isPACKED, msgType, queueing into bitfields.  Inefficient.
 * save  2 bits: coalesce isPACKED with packid. Hard.
 *
 *****************************************************************************/

typedef struct envelope {
  char     core[CmiMsgHeaderSizeBytes];
  
  unsigned int   event;   /* unknown meaning. Used only for logging.*/

  void *     i_tag2;  /* Count OR vidBlockPtr OR chareBlockPtr OR boc_num*/

  unsigned int   TotalSize; /* total size of message, in bytes */

  unsigned short s_tag1;  /* vidPE OR ref OR other_id */
  unsigned short s_tag2;  /* chare_magic_number */

  unsigned short EP;      /* entry point to call */
  unsigned short priosize;/* priority length, measured in bits */

  unsigned short pe;      /* unknown meaning. used only for logging. */
  unsigned char  msgType;
  unsigned char  isPACKED;

  unsigned char  queueing;
  unsigned char  packid;

} ENVELOPE;


/******************************************************************************
 *
 * ChareID Accessors
 *
 *****************************************************************************/

#define GetID_onPE(id) 		        ((id).onPE)
#define SetID_onPE(id,x) 	        ((id).onPE=(x))

#define GetID_chare_magic_number(id)	((id).magic)
#define SetID_chare_magic_number(id,x)	((id).magic=(x))

#define GetID_chareBlockPtr(id)	        ((id).chareBlockPtr)
#define SetID_chareBlockPtr(id,x)       ((id).chareBlockPtr=(x))

/******************************************************************************
 *
 * Charm Envelope Accessors
 *
 * About s_tag1:
 *
 *    other_id is used only for acc, mono, init, tbl msgs 
 *    vidPE is used only if msgType==VidSendOverMsg       
 *    ref is for user messages only.                      
 *
 *****************************************************************************/

#define GetEnv_count(e)		   ((CMK_SIZE_T)(((ENVELOPE *)(e))->i_tag2))
#define SetEnv_count(e,x)	   (((ENVELOPE *)(e))->i_tag2=((void *)(x)))

#define GetEnv_chareBlockPtr(e)	   ((CHARE_BLOCK *)(((ENVELOPE *)(e))->i_tag2))
#define SetEnv_chareBlockPtr(e,x)  (((ENVELOPE *)(e))->i_tag2=((void *)(x)))

#define SetEnv_vidBlockPtr(e,x)	   (((ENVELOPE *)(e))->i_tag2=(x))
#define GetEnv_vidBlockPtr(e)	   ((CHARE_BLOCK *)(((ENVELOPE *)(e))->i_tag2))

#define GetEnv_boc_num(e) 	   ((CMK_SIZE_T)(((ENVELOPE *)(e))->i_tag2))
#define SetEnv_boc_num(e,x) 	   (((ENVELOPE *)(e))->i_tag2=((void *)(x)))

#define GetEnv_other_id(e)   (((ENVELOPE *)(e))->s_tag1)
#define SetEnv_other_id(e,x) (((ENVELOPE *)(e))->s_tag1=(x))

#define GetEnv_vidPE(e)      (((ENVELOPE *)(e))->s_tag1)
#define SetEnv_vidPE(e,x)    (((ENVELOPE *)(e))->s_tag1=(x))

#define GetEnv_ref(e)        (((ENVELOPE *)(e))->s_tag1)
#define SetEnv_ref(e,x)      (((ENVELOPE *)(e))->s_tag1=(x))

#define GetEnv_chare_magic_number(e)	(((ENVELOPE *)(e))->s_tag2)
#define SetEnv_chare_magic_number(e,x)  (((ENVELOPE *)(e))->s_tag2=(x))

#define GetEnv_isPACKED(e)      (((ENVELOPE *)(e))->isPACKED)
#define SetEnv_isPACKED(e,x)    (((ENVELOPE *)(e))->isPACKED=(x))

#define GetEnv_pe(e)		(((ENVELOPE *)(e))->pe)
#define SetEnv_pe(e,x)          (((ENVELOPE *)(e))->pe=(x))

#define GetEnv_event(e)	        (((ENVELOPE *)(e))->event)
#define SetEnv_event(e,x)	(((ENVELOPE *)(e))->event=(x))

#define GetEnv_EP(e) 		(((ENVELOPE *)(e))->EP)
#define SetEnv_EP(e,x) 		(((ENVELOPE *)(e))->EP=(x))

#define GetEnv_queueing(e)      (((ENVELOPE *)(e))->queueing)
#define SetEnv_queueing(e,x)    (((ENVELOPE *)(e))->queueing=(x))

#define GetEnv_priosize(e)      (((ENVELOPE *)(e))->priosize)
#define SetEnv_priosize(e,x)    (((ENVELOPE *)(e))->priosize=(x))

#define GetEnv_TotalSize(e)     (((ENVELOPE *)(e))->TotalSize)
#define SetEnv_TotalSize(e,x)   (((ENVELOPE *)(e))->TotalSize=(x))

#define GetEnv_packid(e)        (((ENVELOPE *)(e))->packid)
#define SetEnv_packid(e,x)      (((ENVELOPE *)(e))->packid=(x))

#define GetEnv_msgType(e)       (((ENVELOPE *)(e))->msgType)
#define SetEnv_msgType(e,x)     (((ENVELOPE *)(e))->msgType=(x))

/******************************************************************************
 *
 * Priority Accessors
 *
 *****************************************************************************/

typedef unsigned int PVECTOR;

#define GetEnv_priowords(e) ((GetEnv_priosize(e)+CINTBITS-1)/CINTBITS)
#define GetEnv_priobytes(e) (GetEnv_priowords(e)*sizeof(int))
#define GetEnv_prioend(e) ((unsigned int *)(((char *)(e))+GetEnv_TotalSize(e)))
#define GetEnv_priobgn(e) ((unsigned int *)(((char *)(e))+GetEnv_TotalSize(e)-GetEnv_priobytes(e)))

extern unsigned int *CkPrioPtrFn       CMK_PROTO((void *));
extern int           CkPrioSizeBitsFn  CMK_PROTO((void *));
extern int           CkPrioSizeBytesFn CMK_PROTO((void *));
extern int           CkPrioSizeWordsFn CMK_PROTO((void *));
extern void          CkPrioConcatFn    CMK_PROTO((void *,void *,unsigned int));

#define CkPrioPtr(msg)       (CkPrioPtrFn((void *)(msg)))
#define CkPrioSizeBits(msg)  (CkPrioSizeBitsFn((void *)(msg)))
#define CkPrioSizeBytes(msg) (CkPrioSizeBytesFn((void *)(msg)))
#define CkPrioSizeWords(msg) (CkPrioSizeWordsFn((void *)(msg)))
#define CkPrioConcat(s,d,x)  (CkPrioConcatFn((void *)(s),(void *)(d),x))


/******************************************************************************
 *
 * Unclassified / Miscellaneous
 *
 *****************************************************************************/

#define CkTraceOn() CpvAccess(traceOn)=1
#define CkTraceOff() CpvAccess(traceOn)=0

#define PACK(x)    CkPack(&x)
#define UNPACK(x)  CkUnpack(&x)

/******************************************************************************
 *
 * THIS SECTION CONSTITUTES THE NEW FORMAT OF A Charm MESSAGE 
 * This file provides access macros for extracting the different
 * sections of a message. The organisation of a message is as follows 
 *
 *           -------------------------------
 *           | env | ldb | user | priority |
 *           -------------------------------
 * 
 *   The sizes of the fields are as follows:
 * 
 *       envelope      : sizeof(ENVELOPE)
 *                        (ENVELOPE is defined in env_macros.h)
 *			First word in ENVELOPE is the core language field.
 * 
 *       ldb           : LDB_ELEM_SIZE is a global variable defined by the
 *                        load balancing module
 * 
 *       user          : the user message data.
 * 
 *       priority      : bit-vector (variable size)
 *
 *   all fields are padded to 8-byte boundaries except the priority,
 *   which is padded to an int-sized boundary.
 *
 * The following variables reflect the message format above. If any
 * change is made to the format, the initialization of the variables must
 * be altered. The variables are initialized in InitializeMessageMacros()
 * in main/common.c. Compile time constants are #defines. 
 * All variables reflect sizes in BYTES.			
 *
 *************************************************************************/


#define ENVELOPE_SIZE sizeof(ENVELOPE)
#define _CK_Env_To_Ldb ENVELOPE_SIZE
#define _CK_Ldb_To_Env (-ENVELOPE_SIZE)

CpvExtern(int,  PAD_SIZE);
CpvExtern(int,  HEADER_SIZE);
CpvExtern(int,  LDB_ELEM_SIZE);
CpvExtern(int, _CK_Env_To_Usr);
CpvExtern(int, _CK_Ldb_To_Usr);
CpvExtern(int, _CK_Usr_To_Env);
CpvExtern(int, _CK_Usr_To_Ldb);

#define TOTAL_MSG_SIZE(usrsize, priowords)\
    (CpvAccess(HEADER_SIZE)+((priowords)*sizeof(int))+(usrsize))


#define LDB_ELEMENT_PTR(env)  \
    (void *) (CHARRED(env) + _CK_Env_To_Ldb)

#define USER_MSG_PTR(env)\
    (CHARRED(env) + CpvAccess(_CK_Env_To_Usr))

#define ENVELOPE_LDBPTR(ldbptr) \
	(ENVELOPE *) (CHARRED(ldbptr) + _CK_Ldb_To_Env)

#define USR_MSG_LDBPTR(ldbptr) \
	(CHARRED(ldbptr) + CpvAccess(_CK_Ldb_To_Usr))

#define ENVELOPE_UPTR(usrptr)\
	(ENVELOPE *) (CHARRED(usrptr) + CpvAccess(_CK_Usr_To_Env))

#define LDB_UPTR(usrptr)\
        (LDB_ELEMENT *) (CHARRED(usrptr) + CpvAccess(_CK_Usr_To_Ldb))

/******************************************************************************
 *
 * BOC Creation Messages and Structures
 *
 *****************************************************************************/

typedef struct {
        int ref;
        int source;
        ChareNumType ep;
        ChareIDType id;
} DYNAMIC_BOC_REQUEST_MSG;

typedef struct {
	ChareNumType boc;
	int ref;
} DYNAMIC_BOC_NUM_MSG;

/******************************************************************************
 *
 * Unclassified / Miscellaneous
 *
 *****************************************************************************/

#define IsCharmPlus(Entry)\
    (CsvAccess(EpInfoTable)[Entry].language==CHARMPLUSPLUS)

#define IsCharmPlusPseudo(id) (CsvAccess(PseudoTable)[id].language==CHARMPLUSPLUS)


#define CkMemError(ptr) if (ptr == NULL) \
                CmiPrintf("*** ERROR *** Memory Allocation Failed --- consider +m command-line option.\n");

#define QDCountThisProcessing(msgType) \
         if ((msgType != QdBocMsg) && (msgType != QdBroadcastBocMsg) && \
			(msgType != LdbMsg)) CpvAccess(msgs_processed)++; 

#define QDCountThisCreation(ep, category, type, x) \
         if ((type != QdBocMsg) && (type != QdBroadcastBocMsg) && \
			(type != LdbMsg)) CpvAccess(msgs_created) += x;


#define ReadValue(v) 			(v)
#define ReadInit(v) 

void *GetBocDataPtr(ChareNumType bocNum);

#define new_packbuffer			CkAllocPackBuffer

#define _CK_4MonoDataAreaType 		MONO_DATA 
#define _CK_9LockMonoDataArea(x)
#define _CK_9GetMonoDataArea		GetBocDataPtr
#define _CK_9UnlockMonoDataArea(x)

#define _CK_4AccDataAreaType 		ACC_DATA
#define _CK_9LockAccDataArea(x)
#define _CK_9GetAccDataArea		GetBocDataPtr
#define _CK_9UnlockAccDataArea(x)

#ifdef STRIP
#define _CK_BroadcastMsgBranch(ep,msg,boc)  GeneralBroadcastMsgBranch(ep,msg,\
					ImmBroadcastBocMsg,boc)
#define _CK_SendMsgBranch(ep,msg,boc,pe)	GeneralSendMsgBranch(ep,msg,pe,\
					ImmBocMsg,boc)
#else
#define _CK_BroadcastMsgBranch(ep,msg,boc)  GeneralBroadcastMsgBranch(ep,msg,\
					BroadcastBocMsg,boc)
#define _CK_SendMsgBranch(ep,msg,boc,pe)	GeneralSendMsgBranch(ep,msg,pe,\
					BocMsg,boc)
#endif
#define _CK_ImmSendMsgBranch(ep,msg,boc,pe)	GeneralSendMsgBranch(ep,msg,pe,\
					ImmBocMsg,boc)

extern void *GenericCkAlloc(int, unsigned int, unsigned int);

CpvExtern(int, _CK_13PackOffset);
CpvExtern(int, _CK_13PackMsgCount);
CpvExtern(int, _CK_13ChareEPCount);
CpvExtern(int, _CK_13TotalMsgCount);
CsvExtern(FUNCTION_PTR*, _CK_9_GlobalFunctionTable);
CsvExtern(MSG_STRUCT*, MsgToStructTable);

typedef int AccIDType;

typedef struct {
  ChareIDType cid;
  EntryPointType EP;	
} ACC_COLLECT_MSG;

FUNCTION_PTR _CK_9GetAccumulateFn();
void * _CK_9_GetAccDataPtr();

typedef int MonoIDType;

#define UP_WAIT_TIME 200
#define MAX_UP_WAIT_TIME 5*200

FUNCTION_PTR _CK_9GetMonoCompareFn();


#define BranchCall(x)		x
#define PrivateCall(x)		x

#if CMK_STATIC_PROTO_WORKS
#define PROTO_PUB_PRIV static
#endif

#if CMK_STATIC_PROTO_FAILS
#define PROTO_PUB_PRIV extern
#endif

#define _CK_Find		TblFind
#define _CK_Delete		TblDelete
#define _CK_Insert		TblInsert
#define _CK_MyBocNum		MyBocNum
#define _CK_CreateBoc		CreateBoc
#define _CK_CreateAcc		CreateAcc
#define _CK_CreateMono		CreateMono
#define _CK_CPlus_CreateAcc	CPlus_CreateAcc
#define _CK_CPlus_CreateMono	CPlus_CreateMono
#define _CK_CreateChare		CreateChare
#define _CK_MyBranchID		MyBranchID
#define _CK_MonoValue		MonoValue

/* charm and charm++ names for converse functions */

#define CK_INT_BITS             (sizeof(int)*8)
#define  C_INT_BITS             (sizeof(int)*8)

#define CkMyPe                  CmiMyPe
#define  CMyPe                  CmiMyPe

#define CkNumPes                 CmiNumPes
#define  CNumPes                 CmiNumPes

#define CkPrintf                CmiPrintf
#define  CPrintf                CmiPrintf

#define CkScanf                 CmiScanf
#define  CScanf                 CmiScanf

#define CkAlloc                 CmiAlloc
#define  CAlloc                 CmiAlloc

#define CkFree                  CmiFree
#define  CFree                  CmiFree

#define CkSpanTreeParent        CmiSpanTreeParent
#define  CSpanTreeParent        CmiSpanTreeParent

#define CkSpanTreeRoot          CmiSpanTreeRoot
#define  CSpanTreeRoot          CmiSpanTreeRoot

#define CkSpanTreeChildren      CmiSpanTreeChildren
#define  CSpanTreeChildren      CmiSpanTreeChildren

#define CkSendToSpanTreeLeaves  CmiSendToSpanTreeLeaves
#define  CSendToSpanTreeLeaves  CmiSendToSpanTreeLeaves

#define CkNumSpanTreeChildren   CmiNumSpanTreeChildren
#define  CNumSpanTreeChildren   CmiNumSpanTreeChildren

#define CK_QUEUEING_FIFO  CQS_QUEUEING_FIFO
#define  C_QUEUEING_FIFO  CQS_QUEUEING_FIFO

#define CK_QUEUEING_LIFO  CQS_QUEUEING_LIFO
#define  C_QUEUEING_LIFO  CQS_QUEUEING_LIFO

#define CK_QUEUEING_IFIFO CQS_QUEUEING_IFIFO
#define  C_QUEUEING_IFIFO CQS_QUEUEING_IFIFO

#define CK_QUEUEING_ILIFO CQS_QUEUEING_ILIFO
#define  C_QUEUEING_ILIFO CQS_QUEUEING_ILIFO

#define CK_QUEUEING_BFIFO CQS_QUEUEING_BFIFO
#define  C_QUEUEING_BLIFO CQS_QUEUEING_BLIFO


/* Charm++ names for charm functions */

#define CPriorityPtr                CkPrioPtr
#define CPrioritySizeBits           CkPrioSizeBits
#define CPrioritySizeBytes          CkPrioSizeBytes
#define CPrioritySizeWords          CkPrioSizeWords
#define CPriorityConcat             CkPrioConcat

#define CStartQuiescence	CPlus_StartQuiescence
#define CharmExit               CkExit
#define CSetQueueing            CkSetQueueing

/* obsolete names */

#define CMyPeNum                CmiMyPe
#define CMaxPeNum               CmiNumPes

#define McMyPeNum() CmiMyPe()
#define McMaxPeNum() CmiNumPes()
#define McTotalNumPe() CmiNumPes()   
#define McSpanTreeInit() CmiSpanTreeInit()
#define McSpanTreeParent(node) CmiSpanTreeParent(node)
#define McSpanTreeRoot() CmiSpanTreeRoot()
#define McSpanTreeChild(node, children) CmiSpanTreeChildren(node, children)
#define McNumSpanTreeChildren(node) CmiNumSpanTreeChildren(node)
#define McSendToSpanTreeLeaves(size, msg) CmiSendToSpanTreeLeaves(size, msg)




/* These are macros for the non-translator version of Charm++.
   They work only for preprocessors with ANSI concatenation, e.g. g++
   So they wont work with cpp.	*/

#define GetEntryPtr(ChareType,EP, MsgType) 	_CK_ep_##ChareType##_##EP##_##MsgType

#define CSendMsg(ChareType,EP,MsgType,msg,ChareId) 	SendMsg(GetEntryPtr(ChareType,EP,MsgType), msg, ChareId)

#define CSendMsgBranch(ChareType,EP,MsgType,msg,ChareId,Pe) 	GeneralSendMsgBranch(GetEntryPtr(ChareType,EP,MsgType), msg, Pe, -1, ChareId)

#define CBroadcastMsgBranch(ChareType,EP,MsgType,msg,ChareId) 	GeneralBroadcastMsgBranch(GetEntryPtr(ChareType,EP,MsgType), msg, -1, ChareId)

#define CLocalBranch(BocType, BocId)	((BocType *)GetBocDataPtr(BocId))

#define CRemoteCallBranch(BOC1, ep1 , mtype, m, g, p) CRemoteCallBranchFn(GetEntryPtr(BOC1,ep1,mtype), m, g, p)

#define CRemoteCall(CH, ep, mtype, m, cid) CRemoteCallFn(GetEntryPtr(CH,ep,mtype),m,cid)

#define MsgIndex(MessageType)	_CK_msg_##MessageType

#define new_chare(ChareType, msgtype, msg)	CreateChare(CMK_CONCAT(_CK_chare_,ChareType), GetEntryPtr(ChareType,ChareType,msgtype), msg, 0, (0xFFF2))

#define new_chare2(ChareType, msgtype, msg, vid, pe) 	CreateChare(CMK_CONCAT(_CK_chare_,ChareType), GetEntryPtr(ChareType,ChareType,msgtype), msg, vid, pe)


#define new_group(ChareType, msgtype, msg)	CreateBoc(CMK_CONCAT(_CK_chare_,ChareType), GetEntryPtr(ChareType,ChareType,msgtype), msg, -1, 0)


#define new_group2(ChareType, msgtype, msg, returnEP, returnID)	CreateBoc(CMK_CONCAT(_CK_chare_,ChareType), GetEntryPtr(ChareType,ChareType,msgtype), msg, returnEP, returnID)

CsvExtern(int, TotalEps);
CpvExtern(int, TotalMsgs);
CpvExtern(int, TotalPseudos);
CpvExtern(int, NumReadMsg);
CpvExtern(int, MsgCount); 		

CpvExtern(int, MainDataSize);  	/* size of dataarea for main chare 	*/
CpvExtern(int, currentBocNum);
CpvExtern(int, InsideDataInit);
CpvExtern(int, mainChare_magic_number);
typedef struct chare_block *CHARE_BLOCK_;
CpvExtern(CHARE_BLOCK_, mainChareBlock);
CpvExtern(CHARE_BLOCK_, currentChareBlock);

CsvExtern(FUNCTION_PTR*, ROCopyFromBufferTable);
CsvExtern(FUNCTION_PTR*, ROCopyToBufferTable);
CsvExtern(EP_STRUCT*, EpInfoTable);
CsvExtern(MSG_STRUCT*, MsgToStructTable); 
CsvExtern(int*,  ChareSizesTable);
CsvExtern(PSEUDO_STRUCT*, PseudoTable);
CsvExtern(char**, EventTable);

CsvExtern(char**, ChareNamesTable);

CpvExtern(int, msgs_processed);
CpvExtern(int, msgs_created);

CpvExtern(int, disable_sys_msgs);
CpvExtern(int, nodecharesProcessed);
CpvExtern(int, nodebocInitProcessed);
CpvExtern(int, nodebocMsgsProcessed);
CpvExtern(int, nodeforCharesProcessed);
CpvExtern(int, nodecharesCreated);
CpvExtern(int, nodeforCharesCreated);
CpvExtern(int, nodebocMsgsCreated);


CpvExtern(int, PrintQueStat); 
CpvExtern(int, PrintMemStat); 
CpvExtern(int, PrintChareStat);
CpvExtern(int, PrintSummaryStat);
CpvExtern(int, QueueingDefault);
CpvExtern(int, CtrLogBufSize);

CpvExtern(int, RecdStatMsg);
CpvExtern(int, CtrRecdTraceMsg);

CpvExtern(int, numHeapEntries);
CpvExtern(int, numCondChkArryElts);

CsvExtern(int, MainChareLanguage);

CpvExtern(int, LDB_ELEM_SIZE);

/* Handlers for various message-types */
CpvExtern(int, HANDLE_INCOMING_MSG_Index);
CsvExtern(int, BUFFER_INCOMING_MSG_Index);
CsvExtern(int, MAIN_HANDLE_INCOMING_MSG_Index);
CsvExtern(int, HANDLE_INIT_MSG_Index);
CpvExtern(int, CkPack_Index);
CpvExtern(int, CkUnpack_Index);
CpvExtern(int, CkInfo_Index);

/* System-defined chare numbers */
CsvExtern(int, CkChare_ACC);
CsvExtern(int, CkChare_MONO);

/* Entry points for Quiescence detection BOC 	*/
CsvExtern(int, CkEp_QD_Init);
CsvExtern(int, CkEp_QD_InsertQuiescenceList);
CsvExtern(int, CkEp_QD_PhaseIBroadcast);
CsvExtern(int, CkEp_QD_PhaseIMsg);
CsvExtern(int, CkEp_QD_PhaseIIBroadcast);
CsvExtern(int, CkEp_QD_PhaseIIMsg);

/* Entry points for Write Once Variables 	*/
CsvExtern(int, CkEp_WOV_AddWOV);
CsvExtern(int, CkEp_WOV_RcvAck);
CsvExtern(int, CkEp_WOV_HostAddWOV);
CsvExtern(int, CkEp_WOV_HostRcvAck);


/* Entry points for dynamic tables BOC    	*/
CsvExtern(int, CkEp_Tbl_Unpack);

/* Entry points for accumulator BOC		*/
CsvExtern(int, CkEp_ACC_CollectFromNode);
CsvExtern(int, CkEp_ACC_LeafNodeCollect);
CsvExtern(int, CkEp_ACC_InteriorNodeCollect);
CsvExtern(int, CkEp_ACC_BranchInit);

/* Entry points for monotonic BOC		*/
CsvExtern(int, CkEp_MONO_BranchInit);
CsvExtern(int, CkEp_MONO_BranchUpdate);
CsvExtern(int, CkEp_MONO_ChildrenUpdate);

/* These are the entry points necessary for the dynamic BOC creation. */
CsvExtern(int, CkEp_DBOC_RegisterDynamicBocInitMsg);
CsvExtern(int, CkEp_DBOC_OtherCreateBoc);
CsvExtern(int, CkEp_DBOC_InitiateDynamicBocBroadcast);

/* These are the entry points for the statistics BOC */
CsvExtern(int, CkEp_Stat_CollectNodes);
CsvExtern(int, CkEp_Stat_Data);
CsvExtern(int, CkEp_Stat_TraceCollectNodes);
CsvExtern(int, CkEp_Stat_BroadcastExitMessage);
CsvExtern(int, CkEp_Stat_ExitMessage);

/* Entry points for LoadBalancing BOC 		*/
CsvExtern(int, CkEp_Ldb_NbrStatus);
CsvExtern(int, NumSysBocEps);

/* Initialization phase count variables for synchronization */
CpvExtern(int,CkInitCount);
CpvExtern(int,CkCountArrived);


/* Buffer for the non-init messages received during the initialization phase */
CpvExtern(void*, CkBuffQueue);

/* Initialization phase flag : 1 if in the initialization phase */
CpvExtern(int, CkInitPhase);

#define MAXMEMSTAT 10 

CpvExtern(int, CstatsMaxChareQueueLength);
CpvExtern(int, CstatsMaxForChareQueueLength);
CpvExtern(int, CstatsMaxFixedChareQueueLength);

typedef struct message3 {
    int srcPE;
    int chareQueueLength;
    int forChareQueueLength;
    int fixedChareQueueLength;
    int charesCreated;
    int charesProcessed;
    int forCharesCreated;
    int forCharesProcessed;
    int bocMsgsCreated;
    int bocMsgsProcessed;
    int nodeMemStat[MAXMEMSTAT];
} STAT_MSG;

typedef FUNCTION_PTR 	*FNPTRTYPE;

#endif





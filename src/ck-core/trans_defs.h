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
 * Revision 2.0  1995-06-02 17:30:04  brunner
 * Reorganized directory structure
 *
 * Revision 1.5  1995/04/13  21:26:44  milind
 * Changed  definition of CkTimer.
 *
 * Revision 1.4  1995/04/13  20:53:46  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.3  1994/12/02  00:02:08  sanjeev
 * interop stuff
 *
 * Revision 1.2  1994/11/11  05:31:10  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:32  brunner
 * Initial revision
 *
 ***************************************************************************/

/* used in EpChareTypeTable, for now this is used only in projections.c */
#define CHARE 		53
#define BOC 		35

/* Used to find which Language an EP is of by EpLanguageTable */
#define CHARM 		0
#define CHARMPLUSPLUS 	1

#define ACCUMULATOR	0
#define MONOTONIC	1
#define TABLE 		2

#define ReadValue(v) 			(v)
#define ReadInit(v) 

#define CkTimer()  			(int)(CmiTimer() * 1000.0)
#define CkUTimer()			(int)(CmiTimer() * 1000000.0)
#define CkHTimer()			(int)(CmiTimer() / 3600.0)

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
					IMMEDIATEcat,BroadcastBocMsg,boc)
#define _CK_SendMsgBranch(ep,msg,boc,pe)	GeneralSendMsgBranch(ep,msg,pe,\
					IMMEDIATEcat,BocMsg,boc)
#else
#define _CK_BroadcastMsgBranch(ep,msg,boc)  GeneralBroadcastMsgBranch(ep,msg,\
					USERcat,BroadcastBocMsg,boc)
#define _CK_SendMsgBranch(ep,msg,boc,pe)	GeneralSendMsgBranch(ep,msg,pe,\
					USERcat,BocMsg,boc)
#endif

#define VOIDFNPTR			FUNCTION_PTR

typedef struct msg_struct {
	int size;
	FUNCTION_PTR pack;
	FUNCTION_PTR unpack;
	FUNCTION_PTR alloc;
} MSG_STRUCT;

typedef struct mono_struct {
	FUNCTION_PTR updatefn;
} MONO_STRUCT;

typedef struct acc_struct {
	FUNCTION_PTR addfn;
	FUNCTION_PTR combinefn;
} ACC_STRUCT;
 
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
		TABLE_STRUCT table;
	} pseudo_type;
} PSEUDO_STRUCT;

/* This causes problems in C++ so its now in trans_externs.h : SANJEEV 
extern void * GenericCkAlloc();
*/

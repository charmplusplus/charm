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
 * Revision 2.21  1998-02-10 21:16:28  milind
 * changed macros to include a msgtype parameter to support method name overloading.
 *
 * Revision 2.20  1997/07/26 16:41:10  jyelon
 * *** empty log message ***
 *
 * Revision 2.19  1996/07/15 21:01:41  jyelon
 * Changed mach-flags from #ifdef to #if
 *
 * Revision 2.18  1996/03/28 21:53:31  sanjeev
 * *** empty log message ***
 *
 * Revision 2.17  1996/03/28 15:25:21  sanjeev
 * *** empty log message ***
 *
 * Revision 2.16  1996/03/22 17:34:11  sanjeev
 * *** empty log message ***
 *
 * Revision 2.15  1996/03/22 17:24:09  sanjeev
 * modified CSendMsgBranch and CBroadcastMsgBranch
 *
 * Revision 2.14  1996/02/22 21:38:20  sanjeev
 * *** empty log message ***
 *
 * Revision 2.13  1996/02/22 21:36:18  sanjeev
 * added macros for non-translator version of Charm++
 *
 * Revision 2.12  1995/11/02 20:23:01  sanjeev
 * removed CFunctionRefToName and CFunctionNameToRef
 *
 * Revision 2.11  1995/11/02  18:24:35  sanjeev
 * modified Charm++ macros
 *
 * Revision 2.10  1995/10/27  21:31:25  jyelon
 * changed NumPe --> NumPes
 *
 * Revision 2.9  1995/10/27  09:09:31  jyelon
 * *** empty log message ***
 *
 * Revision 2.8  1995/10/24  19:48:42  brunner
 * Added Ck* defines for all Mc functions.  For example, McTotalNumPes() ->
 * CkNumPes().
 *
 * Revision 2.7  1995/07/06  22:42:11  narain
 * Changes for LDB interface revision
 *
 * Revision 2.6  1995/07/05  21:17:02  narain
 * Added #defines for timers
 *
 * Revision 2.5  1995/06/14  21:51:24  gursoy
 * *** empty log message ***
 *
 * Revision 2.4  1995/06/14  20:18:55  gursoy
 * *** empty log message ***
 *
 * Revision 2.3  1995/06/13  19:19:07  gursoy
 * CkAlloc and CkFree are added for backward compatibilty
 *
 * Revision 2.2  1995/06/13  14:33:55  gursoy
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/08  17:09:41  gursoy
 * Cpv macro changes done
 *
 * Revision 1.5  1995/04/23  21:21:52  sanjeev
 * added #include converse.h
 *
 * Revision 1.4  1995/04/13  20:53:46  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.3  1994/12/02  00:02:01  sanjeev
 * interop stuff
 *
 * Revision 1.2  1994/11/11  05:31:08  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:30  brunner
 * Initial revision
 *
 ***************************************************************************/

#ifndef CKDEFS_H
#define CKDEFS_H

#include "converse.h"
#include "conv-mach.h"
#include "common.h"
#include "const.h"
#define PROGRAMVARS
#include "trans_defs.h"
#include "trans_decls.h"
#undef PROGRAMVARS
#include "acc.h"
#include "mono.h"
#include "prio_macros.h"
#include "user_macros.h"

#define BranchCall(x)		x
#define PrivateCall(x)		x

#ifndef NULL
#define NULL 0
#endif

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



#endif

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
 * Revision 2.1  1995-06-08 17:09:41  gursoy
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

/* The rest are for use by programs that use the old names */ 
#define McMyPeNum() CmiMyPe()
#define McMaxPeNum() CmiNumPe()
#define McTotalNumPe() CmiNumPe()   
#define McSpanTreeInit() CmiSpanTreeInit()
#define McSpanTreeParent(node) CmiSpanTreeParent(node)
#define McSpanTreeRoot() CmiSpanTreeRoot()
#define McSpanTreeChild(node, children) CmiSpanTreeChildren(node, children)
#define McNumSpanTreeChildren(node) CmiNumSpanTreeChildren(node)
#define McSendToSpanTreeLeaves(size, msg) CmiSendToSpanTreeLeaves(size, msg)

#define CPrintf                 CmiPrintf
#define CScanf                  CmiScanf
#define CMyPeNum                CmiMyPe
#define CharmExit               CkExit

#define CMaxPeNum               CmiNumPe
#define CNumSpanTreeChildren    CmiNumSpanTreeChildren
#define CSpanTreeChild     	CmiSpanTreeChild
#define CSpanTreeParent         CmiSpanTreeParent
#define CSpanTreeRoot           CmiSpanTreeRoot

#define CStartQuiescence	CPlus_StartQuiescence

#define CFunctionNameToRef	FunctionNameToRef
#define CFunctionRefToName	FunctionRefToName

#define CPriorityPtr		PRIORITY_UPTR
#define CPrioritySize		CkPrioritySize

#ifndef NULL
#define NULL 0
#endif



#endif

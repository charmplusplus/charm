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
 * Revision 2.1  1995-06-08 17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.3  1994/12/02  00:02:06  sanjeev
 * interop stuff
 *
 * Revision 1.2  1994/11/11  05:24:04  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:05  brunner
 * Initial revision
 *
 ***************************************************************************/
#ifndef TRANS_DECLS_H
#define TRANS_DECLS_H
CpvExtern(int, _CK_13PackOffset);
CpvExtern(int, _CK_13PackMsgCount);
CpvExtern(int, _CK_13ChareEPCount);
CpvExtern(int, _CK_13TotalMsgCount);
CsvExtern(FUNCTION_PTR*, _CK_9_GlobalFunctionTable);
CsvExtern(MSG_STRUCT*, MsgToStructTable);
#endif

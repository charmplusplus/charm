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
 * Revision 2.1  1998-03-02 14:58:03  jyelon
 * Forgot to check these in last time.
 *
 * Revision 2.1  1995/06/08 17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.2  1994/11/11  05:24:40  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:13  brunner
 * Initial revision
 *
 ***************************************************************************/
/**************************************************************************/
/*                                                                        */
/*      Authors: Wayne Fenton, Balkrishna Ramkumar, Vikram A. Saletore    */
/*                    Amitabh B. Sinha  and  Laxmikant V. Kale            */
/*              (C) Copyright 1990 The Board of Trustees of the           */
/*                          University of Illinois                        */
/*                           All Rights Reserved                          */
/*                                                                        */
/**************************************************************************/

#define TBL_REPLY 1
#define TBL_NOREPLY 2

#define TBL_WAIT_AFTER_FIRST 1
#define TBL_NEVER_WAIT 2
#define TBL_ALWAYS_WAIT 3

message {
	int key;
	char *data;
} TBL_MSG;


#define TblInsert(x1, x2, x3, x4, x5, x6) \
		_CK_Insert(x1, -1, x2, x3, x4, x5, x6, -1)
#define TblDelete(x1, x2, x3, x4, x5)  _CK_Delete(x1, -1, x2, x3, x4, x5)
#define TblFind(x1, x2, x3, x4, x5) _CK_Find(x1, -1, x2, x3, x4, x5)

message {
	int msgs_processed;
} QUIESCENCE_MSG;

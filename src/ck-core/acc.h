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
 * Revision 1.2  1994/11/11  05:24:27  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:09  brunner
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

#ifndef ACC_H
#define ACC_H

typedef int AccIDType;

typedef struct {
	int id;
	int Penum;
	char *dataptr;
	int AlreadyDone;
	ChareIDType CID;
	int NumChildren;
	EntryPointType EP;
} ACC_DATA;

typedef struct {
	ChareIDType cid;
	EntryPointType EP;	
} ACC_COLLECT_MSG;

FUNCTION_PTR _CK_9GetAccumulateFn();
void * _CK_9_GetAccDataPtr();


#endif

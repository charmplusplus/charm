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
 * Revision 1.2  1994/11/11  05:24:34  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:11  brunner
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
#ifndef MONO_H
#define MONO_H
typedef int MonoIDType;

#define UP_WAIT_TIME 200
#define MAX_UP_WAIT_TIME 5*200

typedef struct {
	int id;
	int time;
	char *dataptr;
	int ismodified;
} MONO_DATA;

FUNCTION_PTR _CK_9GetMonoCompareFn();
#endif

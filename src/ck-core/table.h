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
 * Revision 2.0  1995-06-02 17:27:40  brunner
 * Reorganized directory structure
 *
 * Revision 1.2  1994/11/11  05:31:14  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:36  brunner
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

#define TBL_WAITFORDATA 1
#define TBL_NOWAITFORDATA 2

#define TBL_REPLY 1
#define TBL_NOREPLY 2

#define TBL_WAIT_AFTER_FIRST 1
#define TBL_NEVER_WAIT 2
#define TBL_ALWAYS_WAIT 3

#define MAX_TBL_SIZE 211

typedef struct {
	int penum;
	int index;
} map;

typedef struct {
	int key;
	char *data;
} TBL_MSG;


typedef struct address {
	int entry;
	ChareIDType  chareid;
	struct address *next;
} ADDRESS;

typedef struct tbl_element {
	int isDefined;
	int tbl;
	int key;
	char *data;
	int size_data;
	struct address *reply;
	struct tbl_element *next;
}	TBL_ELEMENT;

typedef struct {
	int i;
} DATA_BR_TBL;

typedef struct {
	int i;
} DATA_MNGR_TBL;


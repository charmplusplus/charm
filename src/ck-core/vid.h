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
 * Revision 1.2  1994/11/11  05:25:19  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:24  brunner
 * Initial revision
 *
 ***************************************************************************/
#ifndef VID_H
#define VID_H

typedef struct data_brnch_vid {
	int dummy;
} DATA_BR_VID;

typedef struct dummy_msg {
	int dummy;
} DUMMY_MSG;

typedef struct vid_msg{
 	CHARE_BLOCK *dataPtr;
        int chare_magic_number;
} CHARE_ID_MSG;
#endif

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
 * Revision 1.2  1994/11/11  05:25:19  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:24  brunner
 * Initial revision
 *
 ***************************************************************************/
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

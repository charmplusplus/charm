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
 * Revision 1.3  1995/04/13  20:53:46  sanjeev
 * Changed Mc to Cmi
 *
 * Revision 1.2  1994/11/11  05:31:15  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:38  brunner
 * Initial revision
 *
 ***************************************************************************/

#define CHARE_PROCESSOR(id) id.onPE
#define CHARE_EQUAL(id1, id2) ((id1.onPE == id2.onPE) && \
				(id1.chare_boc.chare_magic_number == \
				 id2.chare_boc.chare_magic_number))
#define CHARE_PRINT(id) CmiPrintf("[%d] chare chars.: pe = %d, magic_number = %d, data_area = 0x%x\n",\
			CmiMyPe(), id.onPE, \
			id.chare_boc.chare_magic_number, \
			id.id_block.chareBlockPtr)

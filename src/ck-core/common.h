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
 * Revision 1.3  1995/04/23  21:23:04  sanjeev
 * added _CK_VARSIZE_UNIT defn
 *
 * Revision 1.2  1994/11/11  05:31:33  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:40:00  brunner
 * Initial revision
 *
 ***************************************************************************/
#ifndef COMMON_H
#define COMMON_H

#define _CK_VARSIZE_UNIT 8

/*** These are the typedefs needed by the user ***/
typedef unsigned int    PVECTOR;

typedef int 		PeNumType;
typedef int 		EntryPointType;
typedef int 		EntryNumType;
typedef int 		ChareNumType;
typedef int 		ChareNameType;
typedef int 		MsgTypes;
typedef int 		MsgCategories;
typedef int 		START_LOOP;
typedef void 		FUNC();
typedef int 		(*FUNCTION_PTR)();   /* defines FUNCTION_PTR as a 
						pointer to functions */
typedef FUNCTION_PTR 	*FNPTRTYPE;
typedef int 		FunctionRefType;
typedef int 		WriteOnceID;    

typedef struct chare_id_type  {
  int tag1; /** This contains: pe, isBOC **/
  union chareboc {
  	int chare_magic_number;
	ChareNumType boc_num;
  } chare_boc;
  union idblock{
  	struct chare_block  * chareBlockPtr;
  	struct vid_block * vidBlockPtr;
  } id_block;
} ChareIDType;


typedef struct boc_block {
  ChareNumType boc_num;             /* boc instance number */
  double dummy;  /* to pad this struct to one word length */
} BOC_BLOCK;


typedef struct chare_block { 
 ChareIDType selfID;
 int dataSize;
 double dummy;
} CHARE_BLOCK ;  

typedef struct vid_block {
  PeNumType vidPenum;
  int chare_magic_number;
  union infoblock {
 	struct fifo_queue * vid_queue;
 	CHARE_BLOCK      * chareBlockPtr;
  } info_block;
} VID_BLOCK;

#endif

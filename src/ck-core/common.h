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
 * Revision 2.5  1995-09-05 22:02:43  sanjeev
 * added chareptr to CHARE_BLOCK
 *
 * Revision 2.4  1995/09/01  02:13:17  jyelon
 * VID_BLOCK, CHARE_BLOCK, BOC_BLOCK consolidated.
 *
 * Revision 2.3  1995/07/22  23:44:13  jyelon
 * *** empty log message ***
 *
 * Revision 2.2  1995/07/12  16:28:45  jyelon
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/08  17:09:41  gursoy
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
  unsigned short        onPE;
  unsigned short        magic;
  struct chare_block   *chareBlockPtr;
} ChareIDType;

typedef struct chare_block { 
 char charekind;                   /* CHAREKIND: CHARE BOCNODE UVID FVID */
 ChareIDType selfID;               /* My chare ID. */
 union {
     ChareNumType boc_num;         /* if a BOC node */
     ChareIDType  realID;          /* if a Filled-VID */
     struct fifo_queue *vid_queue; /* if an Unfilled-VID */
 } x;
 void *chareptr ;		   /* Pointer to the data area of the chare */
 double dummy;                     /* Pad it to 8 bytes */
} CHARE_BLOCK ;  

#endif

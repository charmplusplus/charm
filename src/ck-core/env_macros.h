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
 * Revision 2.12  1997-02-06 19:52:44  jyelon
 * Corrected the core field to take CmiMsgHeaderSizeBytes into account.
 *
 * Revision 2.11  1995/10/27 09:09:31  jyelon
 * *** empty log message ***
 *
 * Revision 2.10  1995/10/13  18:15:53  jyelon
 * K&R changes.
 *
 * Revision 2.9  1995/09/29  09:51:12  jyelon
 * Many small corrections.
 *
 * Revision 2.8  1995/09/01  02:13:17  jyelon
 * VID_BLOCK, CHARE_BLOCK, BOC_BLOCK consolidated.
 *
 * Revision 2.7  1995/07/27  20:29:34  jyelon
 * Improvements to runtime system, general cleanup.
 *
 * Revision 2.6  1995/07/25  00:29:31  jyelon
 * *** empty log message ***
 *
 * Revision 2.5  1995/07/24  01:54:40  jyelon
 * *** empty log message ***
 *
 * Revision 2.4  1995/07/22  23:44:13  jyelon
 * *** empty log message ***
 *
 * Revision 2.3  1995/07/19  22:15:35  jyelon
 * *** empty log message ***
 *
 * Revision 2.2  1995/07/12  16:28:45  jyelon
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.6  1995/05/03  20:58:09  sanjeev
 * *** empty log message ***
 *
 * Revision 1.5  1995/04/23  20:54:43  sanjeev
 * Removed Core....
 *
 * Revision 1.4  1995/03/17  23:37:51  sanjeev
 * changes for better message format
 *
 * Revision 1.3  1995/03/12  17:09:48  sanjeev
 * changes for new msg macros
 *
 * Revision 1.2  1994/11/11  05:31:19  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:42  brunner
 * Initial revision
 *
 ***************************************************************************/
/*****************************************************************/
/** This is the structure of the envelope. The various functions**/
/** that access it follow.					**/
/** This file also contains access macros for the chare-id	**/
/*****************************************************************/
#ifndef ENV_MACROS_H
#define ENV_MACROS_H

#define GetID_onPE(id) 		        ((id).onPE)
#define SetID_onPE(id,x) 	        ((id).onPE=(x))

#define GetID_chare_magic_number(id)	((id).magic)
#define SetID_chare_magic_number(id,x)	((id).magic=(x))

#define GetID_chareBlockPtr(id)	        ((id).chareBlockPtr)
#define SetID_chareBlockPtr(id,x)       ((id).chareBlockPtr=(x))

/*
 * Current envelope size: 256 bits = 32 bytes = 4 doubles.
 *
 * Note: the user-data area is aligned to a 64-bit boundary.  Therefore,
 * there is no point to trimming the envelope unless you can save 64 bits.
 *
 * save 32 bits: remove 'event'.  Easy with ifdefs, doubles SUPER_INSTALL time.
 * save 16 bits: remove 'pe'.     Easy with ifdefs, doubles SUPER_INSTALL time.
 * save 16 bits: change TotalSize to a magnitude.  Inefficient.
 * save 16 bits: could eliminate priosize, by moving it into priority. Clumsy.
 * save  8 bits: remove msgType by replacing HANDLE_X_MSG.  Hard.
 * save 14 bits: turn isPACKED, msgType, queueing into bitfields.  Inefficient.
 * save  2 bits: coalesce isPACKED with packid. Hard.
 *
 */

typedef struct envelope {
  char     core[CmiMsgHeaderSizeBytes];
  
  unsigned int   event;   /* unknown meaning. Used only for logging.*/

  unsigned int   i_tag2;  /* Count OR vidBlockPtr OR chareBlockPtr OR boc_num*/

  unsigned int   TotalSize; /* total size of message, in bytes */

  unsigned short s_tag1;  /* vidPE OR ref OR other_id */
  unsigned short s_tag2;  /* chare_magic_number */

  unsigned short EP;      /* entry point to call */
  unsigned short priosize;/* priority length, measured in bits */

  unsigned short pe;      /* unknown meaning. used only for logging. */
  unsigned char  msgType;
  unsigned char  isPACKED;

  unsigned char  queueing;
  unsigned char  packid;

} ENVELOPE;


#define INTBITS (sizeof(int)*8)

/*********************************************************/
/** Arrangement for i_tag2                              **/
/*********************************************************/
#define GetEnv_count(e)		        (((ENVELOPE *)(e))->i_tag2)
#define SetEnv_count(e,x)		(((ENVELOPE *)(e))->i_tag2=(x))

#define GetEnv_chareBlockPtr(e)	        ((CHARE_BLOCK *)(((ENVELOPE *)(e))->i_tag2))
#define SetEnv_chareBlockPtr(e,x)	(((ENVELOPE *)(e))->i_tag2=((int)(x)))

#define SetEnv_vidBlockPtr(e,x)	        (((ENVELOPE *)(e))->i_tag2=((int)(x)))
#define GetEnv_vidBlockPtr(e)		((CHARE_BLOCK *)(((ENVELOPE *)(e))->i_tag2))

#define GetEnv_boc_num(e) 		(((ENVELOPE *)(e))->i_tag2)
#define SetEnv_boc_num(e,x) 		(((ENVELOPE *)(e))->i_tag2=(x))

/*********************************************************/
/* Arrangement for s_tag1                                */
/* other_id is used only for acc, mono, init, tbl msgs   */
/* vidPE is used only if msgType==VidSendOverMsg         */
/* ref is for user messages only.                        */
/*********************************************************/

#define GetEnv_other_id(e)   (((ENVELOPE *)(e))->s_tag1)
#define SetEnv_other_id(e,x) (((ENVELOPE *)(e))->s_tag1=(x))

#define GetEnv_vidPE(e)      (((ENVELOPE *)(e))->s_tag1)
#define SetEnv_vidPE(e,x)    (((ENVELOPE *)(e))->s_tag1=(x))

#define GetEnv_ref(e)        (((ENVELOPE *)(e))->s_tag1)
#define SetEnv_ref(e,x)      (((ENVELOPE *)(e))->s_tag1=(x))

#define GetEnv_chare_magic_number(e)	(((ENVELOPE *)(e))->s_tag2)
#define SetEnv_chare_magic_number(e,x)  (((ENVELOPE *)(e))->s_tag2=(x))

/*********************************************************/
/** These fields share a byte.                           */
/*********************************************************/

#define GetEnv_isPACKED(e)      (((ENVELOPE *)(e))->isPACKED)
#define SetEnv_isPACKED(e,x)    (((ENVELOPE *)(e))->isPACKED=(x))

/*********************************************************/
/** These fields are alone currently, and accessed	**/
/** separately.						**/
/*********************************************************/

#define GetEnv_pe(e)		(((ENVELOPE *)(e))->pe)
#define SetEnv_pe(e,x)          (((ENVELOPE *)(e))->pe=(x))

#define GetEnv_event(e)	        (((ENVELOPE *)(e))->event)
#define SetEnv_event(e,x)	(((ENVELOPE *)(e))->event=(x))

#define GetEnv_EP(e) 		(((ENVELOPE *)(e))->EP)
#define SetEnv_EP(e,x) 		(((ENVELOPE *)(e))->EP=(x))

#define GetEnv_queueing(e)      (((ENVELOPE *)(e))->queueing)
#define SetEnv_queueing(e,x)    (((ENVELOPE *)(e))->queueing=(x))

#define GetEnv_priosize(e)      (((ENVELOPE *)(e))->priosize)
#define SetEnv_priosize(e,x)    (((ENVELOPE *)(e))->priosize=(x))

#define GetEnv_TotalSize(e)     (((ENVELOPE *)(e))->TotalSize)
#define SetEnv_TotalSize(e,x)   (((ENVELOPE *)(e))->TotalSize=(x))

#define GetEnv_packid(e)        (((ENVELOPE *)(e))->packid)
#define SetEnv_packid(e,x)      (((ENVELOPE *)(e))->packid=(x))

#define GetEnv_msgType(e)       (((ENVELOPE *)(e))->msgType)
#define SetEnv_msgType(e,x)     (((ENVELOPE *)(e))->msgType=(x))

/*********************************/
/* Navigating the priority field */
/*********************************/

#define GetEnv_priowords(e) ((GetEnv_priosize(e)+INTBITS-1)/INTBITS)
#define GetEnv_priobytes(e) (GetEnv_priowords(e)*sizeof(int))
#define GetEnv_prioend(e) ((unsigned int *)(((char *)(e))+GetEnv_TotalSize(e)))
#define GetEnv_priobgn(e) ((unsigned int *)(((char *)(e))+GetEnv_TotalSize(e)-GetEnv_priobytes(e)))

#endif



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
 * Revision 2.4  1995-07-22 23:44:13  jyelon
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

#define GetID_onPE(id) 		((id).onPE)
#define SetID_onPE(id,x) 	((id).onPE=(x))

#define GetID_chare_magic_number(id)	((id).magic)
#define SetID_chare_magic_number(id,x)	((id).magic=(x))

#define GetID_isBOC(id) 	((id).i_tag1&1)
#define GetID_isVID(id) 	((id).i_tag1&2)

#define GetID_boc_num(id)		((id).i_tag1>>2)
#define SetID_boc_num(id,x)             ((id).i_tag1=((x)<<2)|1)

#define GetID_chareBlockPtr(id)		((CHARE_BLOCK *)((id).i_tag1&(~3)))
#define SetID_chareBlockPtr(id,x)       ((id).i_tag1=((int)(x)))

#define GetID_vidBlockPtr(id)		((VID_BLOCK *)((id).i_tag1&(~3)))
#define SetID_vidBlockPtr(id,x)         ((id).i_tag1=((int)(x))|2)

/*
 *
 * 8 bits: pack queueing and msgType into an int.
 * 2 bits: get rid of category & destPeFixed (infer from msgType).
 * 
 * Fields used by most/all messages:
 *
 *   +totsize +event +pe +priosize
 *
 * During CreateChare/VID:
 *
 *   +vidBlockPtr +EP +vidPE +dataMag
 *
 * During CreateChare/no VID:
 *
 *   -vidBlockPtr +EP +vidPE +dataMag
 *
 * During SendMsg/dagger
 *
 *   +chareBlockPtr +EP +ref +magic
 *
 * During SendMsg/no dagger
 *
 *   +chareBlockPtr +EP -ref +magic
 *
 */

typedef struct envelope {
  unsigned int   core1;     /* first word of converse core. */

  unsigned int   event;   /* unknown meaning. Used only for logging.*/
  unsigned int   i_tag2;  /* Count OR vidBlockPtr OR chareBlockPtr OR boc_num*/
  unsigned int   totsize; /* total size of message, in bytes */

  unsigned short EP;      /* entry point to call */
  unsigned short s_tag1;  /* other_id OR ref OR vidPE */

  unsigned short priosize;/* priority length, measured in bits */
  unsigned short pe;      /* unknown meaning. used only for logging. */

  unsigned short magic;   /* dataMag or chare_magic_number */
  unsigned char  c_tag1;  /* category, destPeFixed, isPACKED */
  unsigned char  msgType;

  unsigned char  queueing;
  unsigned char  packid;

} ENVELOPE;


#define env(x) ((ENVELOPE *)(x))
#define INTBITS (sizeof(int)*8)

/*********************************************************/
/** Arrangement for i_tag2                              **/
/** 	count=32bits					**/
/**	vidBlockPtr=32bits				**/
/** 	chareBlockPtr=32bits.				**/
/**	boc_num=32bits.      				**/
/*********************************************************/
#define GetEnv_count(e)		        (env(e)->i_tag2)
#define GetEnv_chareBlockPtr(e)	        (env(e)->i_tag2)
#define GetEnv_vidBlockPtr(e)		(env(e)->i_tag2)
#define GetEnv_boc_num(e) 		(env(e)->i_tag2)

#define SetEnv_count(e,x)		(env(e)->i_tag2=(x))
#define SetEnv_chareBlockPtr(e,x)	(env(e)->i_tag2=((int)(x)))
#define SetEnv_vidBlockPtr(e,x)	        (env(e)->i_tag2=((int)(x)))
#define SetEnv_boc_num(e,x) 		(env(e)->i_tag2=(x))

#define GetEnv_chare_magic_number(e)	(env(e)->magic)
#define SetEnv_chare_magic_number(e,x)  (env(e)->magic=(x))

#define GetEnv_dataMag(e)               (env(e)->magic)
#define SetEnv_dataMag(e,x)             (env(e)->magic=(x))

/*********************************************************/
/** Arrangement for c_tag1				**/
/**  category 1bit | destPeFixed 1bit | isPACKED 2bit   **/
/*********************************************************/

#define GetEnv_category(e)    (env(e)->c_tag1>>7)
#define GetEnv_destPeFixed(e) ((env(e)->c_tag1>>6)&1)
#define GetEnv_isPACKED(e)    ((env(e)->c_tag1>>4)&3)

#define SetEnv_category(e,x)   (env(e)->c_tag1=(env(e)->c_tag1&0x7F)|((x)<<7))
#define SetEnv_destPeFixed(e,x)(env(e)->c_tag1=(env(e)->c_tag1&0xBF)|((x)<<6))
#define SetEnv_isPACKED(e,x)   (env(e)->c_tag1=(env(e)->c_tag1&0xCF)|((x)<<4))

/*********************************************************/
/* other_id is used only for acc, mono, init, tbl msgs   */
/* vidPE is used only if msgType==VidSendOverMsg         */
/* ref is for user messages only.                        */
/*********************************************************/

#define GetEnv_other_id(e)   (env(e)->s_tag1)
#define SetEnv_other_id(e,x) (env(e)->s_tag1=(x))

#define GetEnv_vidPE(e)      (env(e)->s_tag1)
#define SetEnv_vidPE(e,x)    (env(e)->s_tag1=(x))

#define GetEnv_ref(e)        (env(e)->s_tag1)
#define SetEnv_ref(e,x)      (env(e)->s_tag1=(x))

/*********************************************************/
/** These fields are alone currently, and accessed	**/
/** separately.						**/
/*********************************************************/
#define GetEnv_pe(e)			(env(e)->pe)
#define GetEnv_event(e)	        	(env(e)->event)
#define GetEnv_EP(e) 			(env(e)->EP)

#define SetEnv_pe(e,x)		        (env(e)->pe=(x))
#define SetEnv_event(e,x)		(env(e)->event=(x))
#define SetEnv_EP(e,x) 		        (env(e)->EP=(x))

#define GetEnv_queueing(e)    (env(e)->queueing)
#define GetEnv_priosize(e)    (env(e)->priosize)
#define SetEnv_queueing(e,x)  (env(e)->queueing=(x))
#define SetEnv_priosize(e,x)  (env(e)->priosize=(x))

#define SetEnv_prioinfo(e, q, p) (env(e)->queueing=(q),env(e)->priosize=(p))

#define GetEnv_TotalSize(e)             (env(e)->totsize)
#define GetEnv_packid(e)                (env(e)->packid)
#define SetEnv_TotalSize(e,x)           (env(e)->totsize=(x))
#define SetEnv_packid(e,x)              (env(e)->packid=(x))

#define SetEnv_TotalSize_packid(e, sz, id)\
	(env(e)->totsize=(sz),env(e)->packid=(id))

#define GetEnv_msgType(e)    (env(e)->msgType)
#define SetEnv_msgType(e,x)  (env(e)->msgType=(x))

/*********************************/
/* Navigating the priority field */
/*********************************/

#define GetEnv_priowords(e) ((GetEnv_priosize(e)+INTBITS-1)/INTBITS)
#define GetEnv_priobytes(e) (GetEnv_priowords(e)*sizeof(int))
#define GetEnv_prioend(e) (((char *)(e))+GetEnv_TotalSize(e))
#define GetEnv_priobgn(e) (GetEnv_prioend(e)-GetEnv_priobytes(e))

#endif

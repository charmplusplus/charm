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
 * Revision 2.2  1995-07-12 16:28:45  jyelon
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

typedef struct envelope {
  unsigned int core1;
  /* core1 is the first word of the core's fields */

  unsigned int   i_tag1;
  unsigned int   i_tag2;
  unsigned int   i_tag3;
  unsigned int   destPE;

  unsigned short onPE;
  unsigned short ref;

  unsigned short EP;
  unsigned short s_tag1;

  unsigned short priosize;
  unsigned char  queueing;
  unsigned char  c_tag1;
  
#ifdef DEBUGGING_MODE
  int            pe;
  int            event;
#endif 

} ENVELOPE;


#define env(x) ((ENVELOPE *)(x))
#define INTBITS (sizeof(int)*8)

/*********************************************************/
/** 	onPE = 16 bits					**/
/**	isBOC = 8 bits					**/
/**	isBOC overloaded with DynamicBocNum and type	**/
/** 	of chare in case of a send message.		**/
/**	isVID = 8 bits					**/
/**	isVID overloaded with whether a new chare is	**/
/**	created as a vid or not, and the id of an 	**/
/**	accumulator, monotonic and read message 	**/
/** 	variable.					**/ 
/*********************************************************/
#define GetID_onPE(id) 			((id).onPE)
#define GetID_isBOC(id) 		((id).isBOC)
#define GetID_isVID(id) 		((id).isVID)

#define SetID_onPE(id,x) 		((id).onPE=(x))
#define SetID_isBOC(id,x) 		((id).isBOC=(x))
#define SetID_isVID(id,x) 		((id).isVID=(x))

#define GetID_chare_magic_number(id)	id.chare_boc.chare_magic_number
#define GetID_boc_num(id)		id.chare_boc.boc_num
#define GetID_chareBlockPtr(id)		id.id_block.chareBlockPtr
#define GetID_vidBlockPtr(id)		id.id_block.vidBlockPtr

#define SetID_chare_magic_number(id,x)	id.chare_boc.chare_magic_number=x
#define SetID_boc_num(id,x)		id.chare_boc.boc_num=x
#define SetID_chareBlockPtr(id,x)	id.id_block.chareBlockPtr=x
#define SetID_vidBlockPtr(id,x)		id.id_block.vidBlockPtr=x


/*********************************************************/
/** Arrangement for i_tag1                              **/
/**  | totalsize 24bits | packid 8bits |                **/
/*********************************************************/

#define GetEnv_TotalSize(e)             (env(e)->i_tag1>>8)
#define GetEnv_packid(e)                (env(e)->i_tag1&0xFF)

#define SetEnv_TotalSize_packid(e, sz, id) (env(e)->i_tag1 = (sz<<8)|id)

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
#define SetEnv_chareBlockPtr(e,x)	(env(e)->i_tag2=(x))
#define SetEnv_vidBlockPtr(e,x)	        (env(e)->i_tag2=(x))
#define SetEnv_boc_num(e,x) 		(env(e)->i_tag2=(x))

/*********************************************************/
/** Arrangement for i_tag3                              **/
/**	sizeData=32bits					**/
/** 	chare_magic_number=32bits			**/
/*********************************************************/
#define GetEnv_sizeData(e)		(env(e)->i_tag3)
#define GetEnv_chare_magic_number(e)	(env(e)->i_tag3)

#define SetEnv_sizeData(e,x) 	        (env(e)->i_tag3=(x))
#define SetEnv_chare_magic_number(e,x)  (env(e)->i_tag3=(x))

/*********************************************************/
/** Arrangements for s_tag1                         	**/
/**     isVID 2bits | vidEP 14bits                 	**/
/**	other_id 16bits	                                **/
/*********************************************************/

#define GetEnv_isVID(e)	    (env(e)->s_tag1>>14)
#define GetEnv_vidEP(e)     (env(e)->s_tag1&0x3FFF)
#define GetEnv_other_id(e)  (env(e)->s_tag1)

#define SetEnv_isVID(e,x)   (env(e)->s_tag1=(env(e)->s_tag1&0x3FFF)|((x)<<14))
#define SetEnv_vidEP(e,x)   (env(e)->s_tag1=(env(e)->s_tag1&0xC000)|(x))
#define SetEnv_other_id(e,x)(env(e)->s_tag1=(x))

/*********************************************************/
/** Arrangement for c_tag1				**/
/**  category 1bit | destPeFixed 1bit | isPACKED 2bit | msgType 4bit 
/*********************************************************/

#define GetEnv_category(e)    (env(e)->c_tag1>>7)
#define GetEnv_destPeFixed(e) ((env(e)->c_tag1>>6)&1)
#define GetEnv_isPACKED(e)    ((env(e)->c_tag1>>4)&3)
#define GetEnv_msgType(e)     (env(e)->c_tag1&0xF)

#define SetEnv_category(e,x)   (env(e)->c_tag1=(env(e)->c_tag1&0x7F)|((x)<<7))
#define SetEnv_destPeFixed(e,x)(env(e)->c_tag1=(env(e)->c_tag1&0xBF)|((x)<<6))
#define SetEnv_isPACKED(e,x)   (env(e)->c_tag1=(env(e)->c_tag1&0xCF)|((x)<<4))
#define SetEnv_msgType(e,x)    (env(e)->c_tag1=(env(e)->c_tag1&0xF0)|(x))

/*********************************************************/
/** These fields are alone currently, and accessed	**/
/** separately.						**/
/*********************************************************/
#define GetEnv_destPE(e)		(env(e)->destPE)
#define GetEnv_pe(e)			(env(e)->pe)
#define GetEnv_event(e)	        	(env(e)->event)
#define GetEnv_ref(e)                   (env(e)->ref)
#define GetEnv_onPE(e)                  (env(e)->onPE)
#define GetEnv_priosize(e)              (env(e)->priosize)
#define GetEnv_queueing(e)              (env(e)->queueing)
#define GetEnv_EP(e) 			(env(e)->EP)

#define SetEnv_destPE(e,x) 		(env(e)->destPE=(x))
#define SetEnv_pe(e,x)		        (env(e)->pe=(x))
#define SetEnv_event(e,x)		(env(e)->event=(x))
#define SetEnv_ref(e,x)                 (env(e)->ref=(x))
#define SetEnv_onPE(e,x)                (env(e)->onPE=(x))
#define SetEnv_priosize(e,x)            (env(e)->priosize=(x))
#define SetEnv_queueing(e,x)            (env(e)->queueing=(x))
#define SetEnv_EP(e,x) 		        (env(e)->EP=(x))

/*********************************/
/* Navigating the priority field */
/*********************************/

#define GetEnv_priowords(e) ((env(e)->priosize+INTBITS-1)/INTBITS)
#define GetEnv_priobytes(e) (GetEnv_priowords(e)*sizeof(int))
#define GetEnv_prioend(e) (((char *)(e))+GetEnv_TotalSize(e))
#define GetEnv_priobgn(e) (GetEnv_prioend(e)-GetEnv_priobytes(e))

#endif

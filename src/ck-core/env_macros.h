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

#ifdef DEBUGGING_MODE
typedef struct envelope {
  int		core1;
  /* core1 is the first word of the core's fields */

  int		prio;
  /* prio contains 1 bit for priotype, 31 bits for signed integer priority,
     or offset + size for bitvector priority */

  int           tag1;
  /** tag1 contains: ref, msgType, isPACKED, packid  **/

  int           tag2;
  /** tag2 contains: EP, isVID+vidEP / other_id **/

  int           tag3;
  /** tag3 contains: count, vidBlockPtr, chareBlockPtr, boc_id+boc_num  **/

  int           tag4;
  /** tag4 contains: chare_magic_number, dataSize **/

  int           tag5;
  /** tag5 contains: onPE **/

  int           destPE;
  int           pe;
  int           event;
} ENVELOPE;

#else
typedef struct envelope {
  int		core1;
  /* core1 is the first word of the core's fields */

  int		prio;
  /* prio contains 1 bit for priotype, 31 bits for signed integer priority,
     or offset + size for bitvector priority */

  int           tag1;
  /** tag1 contains: ref, category, msgType, isPACKED, packid, destPeFixed; **/

  int           tag2;
  /** tag2 contains: EP, isVID+vidEP / other_id **/

  int           tag3;
  /** tag3 contains: count, vidBlockPtr, chareBlockPtr, boc_id+boc_num  **/

  int           tag4;
  /** tag4 contains: chare_magic_number, dataSize **/

  int           tag5;
  /** tag5 contains: onPE **/

  int           destPE;
} ENVELOPE;
#endif

/*********************************************************/
/** We are trying to set the most bits of x to y.	**/
/*********************************************************/
#define MOST16				0xffff0000
#define zeroMOST16(x,p,n)		(x & ~MOST16)
#define setbitsMOST16(x,y,p,n)		x = (zeroMOST16(x,p,n) | (y << (p+1-n)))

/*********************************************************/
/** We are trying to set the least 16 bits of x to y.	**/
/*********************************************************/
#define LEAST16				0xffff
#define zeroLEAST16(x,p,n)		(x & ~LEAST16)
#define setbitsLEAST16(x,y,p,n)		x = (zeroLEAST16(x,p,n) | (y << (p+1-n)))


/*********************************************************/
/** We are trying to set the next 8 bits of x to y.	**/
/*********************************************************/
#define MIDDLE8				0xff00
#define zeroMIDDLE8(x,p,n)		(x & ~MIDDLE8)
#define setbitsMIDDLE8(x,y,p,n)		x = (zeroMIDDLE8(x,p,n) | (y << (p+1-n)))

/*********************************************************/
/** We are trying to set the least 8 bits of x to y.	**/
/*********************************************************/
#define LEAST8				0xff
#define zeroLEAST8(x,p,n)		(x & ~LEAST8)
#define setbitsLEAST8(x,y,p,n)		x = (zeroLEAST8(x,p,n) | (y << (p+1-n)))

/*********************************************************/
/** We are trying to set bits 16-15 of x to y.		**/
/*********************************************************/
#define NEXT2				0xc000
#define zeroNEXT2(x,p,n)		(x & ~NEXT2 )
#define setbitsNEXT2(x,y,p,n)		x = (zeroNEXT2(x,p,n) | (y << (p+1-n)))

/*********************************************************/
/** We are trying to set bits 14-1 of x to y.		**/
/*********************************************************/
#define NEXT14				0x3fff
#define zeroNEXT14(x,p,n)		(x & ~NEXT14 )
#define setbitsNEXT14(x,y,p,n)		x = (zeroNEXT14(x,p,n) | (y << (p+1-n)))

/*********************************************************/
/** We are trying to set bits 14-7 of x to y.		**/
/*********************************************************/
#define NEXT8				0x3fc0
#define zeroNEXT8(x,p,n)		(x & ~NEXT8 )
#define setbitsNEXT8(x,y,p,n)		x = (zeroNEXT8(x,p,n) | (y << (p+1-n)))

/*********************************************************/
/** We are trying to set the sixth bit of x to y.	**/
/*********************************************************/
#define SIXTH	 			0x20
#define zeroSIXTH(x,p,n)		(x & ~SIXTH )
#define setbitsSIXTH(x,y,p,n)		x = (zeroSIXTH(x,p,n) | (y << (p+1-n)))

/*********************************************************/
/** We are trying to set bits 2-5 of x to y.		**/
/*********************************************************/
#define MIDDLE4	 			0x1e
#define zeroMIDDLE4(x,p,n)		(x & ~MIDDLE4 )
#define setbitsMIDDLE4(x,y,p,n)		x = (zeroMIDDLE4(x,p,n) | (y << (p+1-n)))

/*********************************************************/
/** We are trying to set the first bit of x to y.	**/
/*********************************************************/
#define FIRST	 			0x1
#define zeroFIRST(x,p,n)		(x & ~FIRST )
#define setbitsFIRST(x,y,p,n)		x = (zeroFIRST(x,p,n) | (y << (p+1-n)))

#define getbits(x,p,n) 			((x >> (p+1-n)) & ~(~0 << n))


/*********************************************************/
/** Arrangement for the first tag word in ChareIDType:	**/
/** 	pe = 16 bits					**/
/**	isBOC = 8 bits					**/
/**	isBOC overloaded with DynamicBocNum and type	**/
/** 	of chare in case of a send message.		**/
/**	isVID = 8 bits					**/
/**	isVID overloaded with whether a new chare is	**/
/**	created as a vid or not, and the id of an 	**/
/**	accumulator, monotonic and read message 	**/
/** 	variable.					**/ 
/*********************************************************/
#define GetID_onPE(id) 			getbits(id.tag1,31,16)
#define GetID_isBOC(id) 		getbits(id.tag1,15,8)
#define GetID_isVID(id) 		getbits(id.tag1,7,8)

#define SetID_onPE(id,x) 		setbitsMOST16(id.tag1,x,31,16)
#define SetID_isBOC(id,x) 		setbitsMIDDLE8(id.tag1,x,15,8)
#define SetID_isVID(id,x) 		setbitsLEAST8(id.tag1,x,7,8)

#define GetID_chare_magic_number(id)	id.chare_boc.chare_magic_number
#define GetID_boc_num(id)		id.chare_boc.boc_num
#define GetID_chareBlockPtr(id)		id.id_block.chareBlockPtr
#define GetID_vidBlockPtr(id)		id.id_block.vidBlockPtr

#define SetID_chare_magic_number(id,x)	id.chare_boc.chare_magic_number=x
#define SetID_boc_num(id,x)		id.chare_boc.boc_num=x
#define SetID_chareBlockPtr(id,x)	id.id_block.chareBlockPtr=x
#define SetID_vidBlockPtr(id,x)		id.id_block.vidBlockPtr=x


/*********************************************************/
/** Arrangement for prio word in the envelope:	        
    | signed integer prio : 31 bits | type : 1 bit |
    | size : 11 bits | offset : 20 bits | type : 1 bit |
**********************************************************/
#define GetEnv_PrioType(env)  \
        (int)(((ENVELOPE *)(env))->prio & 0x00000001)
 
#define SetEnv_PrioType(env,x)  \
        ((ENVELOPE *)(env))->prio = x

#define GetEnv_IntegerPrio(env)  \
        (int)(((ENVELOPE *)(env))->prio/2)
 
#define SetEnv_IntegerPrio(env,x)  \
        ((ENVELOPE *)(env))->prio = 2*(x) | (((ENVELOPE *)(env))->prio & 0x00000001) 
 
#define GetEnv_PrioOffset(env)  \
        (int)( (((ENVELOPE *)(env))->prio>>1) & 0x000fffff )
 
#define SetEnv_PrioOffset(env,x)  \
        ((ENVELOPE *)(env))->prio = ( (((ENVELOPE *)(env))->prio & 0xffe00001) | (x<<1) )
 
#define GetEnv_PrioSize(env)  \
        (int)( ((ENVELOPE *)(env))->prio >> 21 )

#define SetEnv_PrioSize(env,x)  \
        ((ENVELOPE *)(env))->prio = ( (((ENVELOPE *)(env))->prio & 0x001fffff) | (x<<21) )


/**  EXTRA MACROS FOR PRIORITY FIELDS  *********************************/

#define GetEnv_PriorityPtr(env, priorityptr) {\
    if ( GetEnv_PrioType((env)) == 0 ) \
        priorityptr = (PVECTOR *)((char *)(env) + 4) ; \
    else  \
        priorityptr = (PVECTOR *)((char *)(env) + *((int *)GetEnv_PrioOffset(env)));\
}
 
#define ReturnEnv_PriorityPtr(env) \
    ( ( GetEnv_PrioType(env) == 0 ) ?  \
      ( ((char *)(env)) + 4 ) : \
      ( ((char *)(env)) + GetEnv_PrioOffset(env) )  \
    )
 




/*********************************************************/
/** Arrangement for first tag word in the envelope:	**/
/** 	ref = 16 bits					**/
/** 	isPACKED = 2 bits 				**/
/** 	packid = 8 bits 				**/
/** 	category = 1 bit 	          		**/
/** 	msgType = 4 bits 				**/
/** 	destPeFixed = 1 bit 	          		**/
/*********************************************************/
#define GetEnv_ref(env) 		getbits(env->tag1,31,16)
#define GetEnv_isPACKED(env) 		getbits(env->tag1,15,2)
#define GetEnv_packid(env) 		getbits(env->tag1,13,8)
#define GetEnv_category(env) 		getbits(env->tag1,5,1)
#define GetEnv_msgType(env) 		getbits(env->tag1,4,4)
#define GetEnv_destPeFixed(env) 	getbits(env->tag1,0,1)

#define SetEnv_ref(env,x) 		setbitsMOST16(env->tag1,x,31,16)
#define SetEnv_isPACKED(env,x) 		setbitsNEXT2(env->tag1,x,15,2)
#define SetEnv_packid(env,x) 		setbitsNEXT8(env->tag1,x,13,8)
#define SetEnv_category(env,x)		setbitsSIXTH(env->tag1,x,5,1)
#define SetEnv_msgType(env,x) 		setbitsMIDDLE4(env->tag1,x,4,4)
#define SetEnv_destPeFixed(env,x) 	setbitsFIRST(env->tag1,x,0,1)

/*********************************************************/
/** Arrangement for second tag word in the envelope:	**/
/** 	EP = 16 bits					**/
/**	isVID=2bits+vidEP=14bits / other_id=16bits	**/
/*********************************************************/
#define GetEnv_EP(env) 			getbits(env->tag2,31,16)
#define GetEnv_isVID(env)		getbits(env->tag2,15,2)
#define GetEnv_vidEP(env) 		getbits(env->tag2,13,14)
#define GetEnv_other_id(env) 		getbits(env->tag2,15,16)

#define SetEnv_EP(env,x) 		setbitsMOST16(env->tag2,x,31,16)
#define SetEnv_isVID(env,x) 		setbitsNEXT2(env->tag2,x,15,2)
#define SetEnv_vidEP(env,x) 		setbitsNEXT14(env->tag2,x,13,14)
#define SetEnv_other_id(env,x) 		setbitsLEAST16(env->tag2,x,15,16)


/*********************************************************/
/** Arrangement for third tag word in the envelope:	**/
/** 	count=32bits					**/
/**	vidBlockPtr=32bits				**/
/** 	chareBlockPtr=32bits.				**/
/**	boc_id=16bits+boc_num=16bits			**/
/*********************************************************/
#define GetEnv_count(env)		env->tag3
#define GetEnv_chareBlockPtr(env)	env->tag3
#define GetEnv_vidBlockPtr(env)		env->tag3
#define GetEnv_boc_id(env)		getbits(env->tag3,31,16)
#define GetEnv_boc_num(env) 		getbits(env->tag3,15,16)

#define SetEnv_count(env,x)		env->tag3=x
#define SetEnv_chareBlockPtr(env,x)	env->tag3=x
#define SetEnv_vidBlockPtr(env,x)	env->tag3=x
#define SetEnv_boc_id(env,x) 		setbitsMOST16(env->tag3,x,31,16)
#define SetEnv_boc_num(env,x) 		setbitsLEAST16(env->tag3,x,15,16)


/*********************************************************/
/** Arrangement for fourth tag word in the envelope:	**/
/**	sizeData=32bits					**/
/** 	chare_magic_number=32bits			**/
/*********************************************************/
#define GetEnv_sizeData(env)		env->tag4
#define GetEnv_chare_magic_number(env)	env->tag4

#define SetEnv_sizeData(env,x) 		env->tag4=x
#define SetEnv_chare_magic_number(env,x) env->tag4=x


/*********************************************************/
/** Arrangement for fifth tag word in the envelope:	**/
/** 	onPE = 32 bits					**/
/*********************************************************/
#define GetEnv_onPE(env) 		env->tag5
#define SetEnv_onPE(env,x) 		env->tag5=x


/*********************************************************/
/** These fields are alone currently, and accessed	**/
/** separately.						**/
/*********************************************************/
#define GetEnv_destPE(env)		env->destPE
#define GetEnv_pe(env)			env->pe
#define GetEnv_event(env)		env->event

#define SetEnv_destPE(env,x) 		env->destPE=x
#define SetEnv_pe(env,x)		env->pe=x
#define SetEnv_event(env,x)		env->event=x

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
 * Revision 2.2  1995-10-27 09:09:31  jyelon
 * *** empty log message ***
 *
 * Revision 2.1  1995/06/08  17:07:12  gursoy
 * Cpv macro changes done
 *
 * Revision 1.2  1994/11/11  05:25:06  brunner
 * Removed ident added by accident with RCS header
 *
 * Revision 1.1  1994/11/07  15:39:21  brunner
 * Initial revision
 *
 ***************************************************************************/

#ifndef _PRIO_MACROS_H
#define _PRIO_MACROS_H

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned int *CkPrioPtrFn       CMK_PROTO((void *));
extern int           CkPrioSizeBitsFn  CMK_PROTO((void *));
extern int           CkPrioSizeBytesFn CMK_PROTO((void *));
extern int           CkPrioSizeWordsFn CMK_PROTO((void *));
extern void          CkPrioConcatFn    CMK_PROTO((void *, void *, unsigned int));

#define CkPrioPtr(msg)       (CkPrioPtrFn((void *)(msg)))
#define CkPrioSizeBits(msg)  (CkPrioSizeBitsFn((void *)(msg)))
#define CkPrioSizeBytes(msg) (CkPrioSizeBytesFn((void *)(msg)))
#define CkPrioSizeWords(msg) (CkPrioSizeWordsFn((void *)(msg)))
#define CkPrioConcat(s,d,x)  (CkPrioConcatFn((void *)(s),(void *)(d),x))

#ifdef __cplusplus
}
#endif

#endif /* _PRIO_MACROS_H */

/*
 * QuickThreads -- Threads-building toolkit.
 * Copyright (c) 1993 by David Keppel
 *
 * Permission to use, copy, modify and distribute this software and
 * its documentation for any purpose and without fee is hereby
 * granted, provided that the above copyright notice and this notice
 * appear in all copies.  This software is provided as a
 * proof-of-concept and for demonstration purposes; there is no
 * representation about the suitability of this software for any
 * purpose.
 */

#ifndef QT_M88K_H
#define QT_M88K_H

typedef unsigned long qt_word_t;

#define QT_GROW_DOWN

/* Stack layout on the mips:

   Callee-save registers are: $16-$23, $30; $f20-$f30.
   Also save $31, return pc.

   Non-varargs:

   +---
   | r30 (fp)	on startup === 0
   | r25
   | r24
   | r23
   | r22
   | r21
   | r20
   | r19
   | r18
   | r17	on startup === `only'
   | r16	on startup === `userf'
   | r15	on startup === `pt'
   | r14	on startup === `pu'
   | r1		on startup === `qt_start'
   | 0
   | 0
   +---
   | 0
   | ... (8 regs worth === 32 bytes of homing area)
   | 0						<--- sp
   +---

   Conventions for varargs:

   |  :
   | arg8
   +---
   | r30 (fp)	arg7
   | r25	arg6
   | r24	arg5
   | r23	arg4
   | r22	arg3
   | r21	arg2
   | r20	arg1
   | r19	arg0
   | r18
   | r17	on startup === `startup'
   | r16	on startup === `vuserf'
   | r15	on startup === `pt'
   | r14	on startup === `cleanup'
   | r1		on startup === `qt_vstart'
   | 0
   | 0
   +---
   | 0
   | ... (8 regs worth === 32 bytes of homing area)
   | 0						<--- sp
   +---

   */


/* Stack must be doubleword aligned. */
#define QT_STKALIGN	(16)	/* Doubleword aligned. */

/* How much space is allocated to hold all the crud for
   initialization: saved registers plus padding to keep the stack
   aligned plus 8 words of padding to use as a `homing area' (for
   r2-r9) when calling helper functions on the stack of the (not yet
   started) thread.  The varargs save area is small because it gets
   overlapped with the top of the parameter list.  In case the
   parameter list is less than 8 args, QT_ARGS_MD0 adds some dead
   space at the top of the stack. */

#define QT_STKBASE	(16*4 + 8*4)
#define QT_VSTKBASE	(8*4 + 8*4)


/* Index of various registers. */
#define QT_1	(8+2)
#define QT_14	(8+3)
#define QT_15	(8+4)
#define QT_16	(8+5)
#define QT_17	(8+6)
#define QT_30	(8+15)


/* When a never-before-run thread is restored, the return pc points
   to a fragment of code that starts the thread running.  For
   non-vargs functions, it sets up arguments and calls the client's
   `only' function.  For varargs functions, the startup code calls the
   startup, user, and cleanup functions.

   For non-varargs functions, we set the frame pointer to 0 to
   null-terminate the call chain.

   For varargs functions, the frame pointer register is used to hold
   one of the arguments, so that all arguments can be laid out in
   memory by the conventional `qt_vargs' varargs initialization
   routine.

   The varargs startup routine always reads 8 words of arguments from
   the stack.  If there are less than 8 words of arguments, then the
   arg list could call off the top of the stack.  To prevent fall-off,
   always allocate 8 words. */

extern void qt_start(void);
#define QT_ARGS_MD(sp) \
  (QT_SPUT (sp, QT_1, qt_start), \
   QT_SPUT (sp, QT_30, 0))


/* The m88k uses a struct for `va_list', so pass a pointer to the
   struct. */

typedef void (qt_function_t)(void);

struct qt_t;
extern struct qt_t *qt_vargs (struct qt_t *sp, int nbytes,
			      void *vargs, void *pt,
			      qt_function_t *startup,
			      qt_function_t *vuserf,
			      qt_function_t *cleanup);

#define QT_VARGS(sp, nbytes, vargs, pt, startup, vuserf, cleanup) \
  (qt_vargs (sp, nbytes, &(vargs), pt, (qt_function_t *)startup, \
	     (qt_function_t *)vuserf, (qt_function_t *)cleanup))


/* The *index* (positive offset) of where to put each value. */
#define QT_ONLY_INDEX	(QT_17)
#define QT_USER_INDEX	(QT_16)
#define QT_ARGT_INDEX	(QT_15)
#define QT_ARGU_INDEX	(QT_14)

#define QT_VCLEANUP_INDEX	(QT_14)
#define QT_VUSERF_INDEX		(QT_16)
#define QT_VSTARTUP_INDEX	(QT_17)
#define QT_VARGT_INDEX		(QT_15)

#endif /* ndef QT_M88K_H */

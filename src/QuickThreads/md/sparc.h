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

#ifndef QT_SPARC_H
#define QT_SPARC_H

typedef unsigned long qt_word_t;

/* Stack layout on the sparc:

   non-varargs:

   +---
   | <blank space for alignment>
   | %o7 == return address -> qt_start
   | %i7
   | %i6 == frame pointer -> 0 (NULL-terminated stack frame chain)
   | %i5 -> only
   | %i4 -> userf
   | %i3
   | %i2 -> pt
   | %i1 -> pu
   | %i0
   | %l7
   | %l6
   | %l5
   | %l4
   | %l3
   | %l2
   | %l1
   | %l0	<--- qt_t.sp
   +---

   varargs:

   |  :
   |  :
   | argument list
   | one-word aggregate return pointer
   +---
   | <blank space for alignment>
   | %o7 == return address -> qt_vstart
   | %i7
   | %i6 == frame pointer -> 0 (NULL-terminated stack frame chain)
   | %i5 -> startup
   | %i4 -> userf
   | %i3 -> cleanup
   | %i2 -> pt
   | %i1
   | %i0
   | %l7
   | %l6
   | %l5
   | %l4
   | %l3
   | %l2
   | %l1
   | %l0	<--- qt_t.sp
   +---

   */


/* What to do to start a thread running. */
extern void qt_start (void);
extern void qt_vstart (void);


/* Hold 17 saved registers + 1 word for alignment. */
#define QT_STKBASE	(18 * 4)
#define QT_VSTKBASE	QT_STKBASE


/* Stack must be doubleword aligned. */
#define QT_STKALIGN	(8)	/* Doubleword aligned. */

#define QT_ONLY_INDEX	(QT_I5)
#define QT_USER_INDEX	(QT_I4)
#define QT_ARGT_INDEX	(QT_I2)
#define QT_ARGU_INDEX	(QT_I1)

#define QT_VSTARTUP_INDEX	(QT_I5)
#define QT_VUSERF_INDEX		(QT_I4)
#define QT_VCLEANUP_INDEX	(QT_I3)
#define QT_VARGT_INDEX		(QT_I2)

#define QT_O7	(16)
#define QT_I6	(14)
#define QT_I5	(13)
#define QT_I4	(12)
#define QT_I3	(11)
#define QT_I2	(10)
#define QT_I1	( 9)


/* The thread will ``return'' to the `qt_start' routine to get things
   going.  The normal return sequence takes us to QT_O7+8, so we
   pre-subtract 8.  The frame pointer chain is 0-terminated to prevent
   the trap handler from chasing off in to random memory when flushing
   stack windows. */

#define QT_ARGS_MD(top) \
    (QT_SPUT ((top), QT_O7, ((void *)(((int)qt_start)-8))), \
     QT_SPUT ((top), QT_I6, 0))


/* The varargs startup routine always reads 6 words of arguments
   (6 argument registers) from the stack, offset by one word to
   allow for an aggregate return area  pointer.  If the varargs
   routine actually pushed fewer words than that, qt_vstart could read
   off the top of the stack.  To prevent errors, we always allocate 8
   words.  The space is often just wasted. */

#define QT_VARGS_MD0(sp, vabytes) \
  ((qt_t *)(((char *)(sp)) - 8*4 - QT_STKROUNDUP(vabytes)))

#define QT_VARGS_MD1(sp) \
  (QT_SPUT (sp, QT_O7, ((void *)(((int)qt_vstart)-8))))

/* The SPARC has wierdo calling conventions which stores a hidden
   parameter for returning aggregate values, so the rest of the
   parameters are shoved up the stack by one place. */
#define QT_VARGS_ADJUST(sp)	(((char *)sp)+4)

#define QT_VARGS_DEFAULT


#define QT_GROW_DOWN

#endif /* ndef QT_SPARC_H */

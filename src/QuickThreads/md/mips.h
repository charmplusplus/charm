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

#ifndef QT_MIPS_H
#define QT_MIPS_H

typedef unsigned long qt_word_t;

#define QT_GROW_DOWN

/* Stack layout on the mips:

   Callee-save registers are: $16-$23, $30; $f20-$f30.
   Also save $31, return pc.

   Non-varargs:

   +---
   | $f30	The first clump is only saved if `qt_block'
   | $f28	is called, in which case it saves the fp regs
   | $f26	then calls `qt_blocki' to save the int regs.
   | $f24
   | $f22
   | $f20
   | $31 === return pc in `qt_block'
   +---
   | $31 === return pc; on startup == qt_start
   | $30
   | $23
   | $22
   | $21
   | $20
   | $19	on startup === only
   | $18	on startup === $a2 === userf
   | $17	on startup === $a1 === pt
   | $16	on startup === $a0 === pu
   | <a3>	save area req'd by MIPS calling convention
   | <a2>	save area req'd by MIPS calling convention
   | <a1>	save area req'd by MIPS calling convention
   | <a0>	save area req'd by MIPS calling convention	<--- sp
   +---

   Conventions for varargs:

   | args ...
   +---
   |  :
   |  :
   | $21
   | $20
   | $19	on startup === `userf'
   | $18	on startup === `startup'
   | $17	on startup === `pt'
   | $16	on startup === `cleanup'
   | <a3>
   | <a2>
   | <a1>
   | <a0>	<--- sp
   +---

   Note: if we wanted to, we could muck about and try to get the 4
   argument registers loaded in to, e.g., $22, $23, $30, and $31,
   and the return pc in, say, $20.  Then, the first 4 args would
   not need to be loaded from memory, they could just use
   register-to-register copies. */


/* Stack must be doubleword aligned. */
#define QT_STKALIGN	(8)	/* Doubleword aligned. */

/* How much space is allocated to hold all the crud for
   initialization: $16-$23, $30, $31.  Just do an integer restore,
   no need to restore floating-point.  Four words are needed for the
   argument save area for the helper function that will be called for
   the old thread, just before the new thread starts to run. */

#define QT_STKBASE	(14 * 4)
#define QT_VSTKBASE	QT_STKBASE


/* Offsets of various registers. */
#define QT_31	(9+4)
#define QT_19	(3+4)
#define QT_18	(2+4)
#define QT_17	(1+4)
#define QT_16	(0+4)


/* When a never-before-run thread is restored, the return pc points
   to a fragment of code that starts the thread running.  For
   non-vargs functions, it just calls the client's `only' function.
   For varargs functions, it calls the startup, user, and cleanup
   functions.

   The varargs startup routine always reads 4 words of arguments from
   the stack.  If there are less than 4 words of arguments, then the
   startup routine can read off the top of the stack.  To prevent
   errors we always allocate 4 words.  If there are more than 3 words
   of arguments, the 4 preallocated words are simply wasted. */

extern void qt_start(void);
#define QT_ARGS_MD(sp)	(QT_SPUT (sp, QT_31, qt_start))

#define QT_VARGS_MD0(sp, vabytes) \
  ((qt_t *)(((char *)(sp)) - 4*4 - QT_STKROUNDUP(vabytes)))

extern void qt_vstart(void);
#define QT_VARGS_MD1(sp)	(QT_SPUT (sp, QT_31, qt_vstart))

#define QT_VARGS_DEFAULT


/* The *index* (positive offset) of where to put each value. */
#define QT_ONLY_INDEX	(QT_19)
#define QT_USER_INDEX	(QT_18)
#define QT_ARGT_INDEX	(QT_17)
#define QT_ARGU_INDEX	(QT_16)

#define QT_VCLEANUP_INDEX	(QT_16)
#define QT_VUSERF_INDEX		(QT_19)
#define QT_VSTARTUP_INDEX	(QT_18)
#define QT_VARGT_INDEX		(QT_17)

#endif /* ndef QT_MIPS_H */

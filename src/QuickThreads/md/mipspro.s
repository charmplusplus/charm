/* mips.s -- assembly support. */

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

/* Callee-save $16-$23, $30-$31.
 *
 * $25 is used as a procedure value pointer, used to discover constants
 * in a callee.  Thus, each caller here sets $25 before the call.
 *
 * On startup, restore regs so retpc === call to a function to start.
 * We're going to call a function ($4) from within this routine.
 * We're passing 3 args, therefore need to allocate 12 extra bytes on
 * the stack for a save area.  The start routine needs a like 16-byte
 * save area.  Must be doubleword aligned (_mips r3000 risc
 * architecture_, gerry kane, pg d-23).
 */

/*
 * Modified by Assar Westerlund <assar@sics.se> to support Irix 5.x
 * calling conventions for dynamically-linked code.
 */

	.set noreorder
	/* Make this position-independent code. */
	.option pic2

	.globl qt_block
	.globl qt_blocki
	.globl qt_abort
	.globl qt_start
	.globl qt_vstart

	/*
	** $4: ptr to function to call once curr is suspended
	**	and control is on $7's stack.
	** $5: 1'th arg to $4.
	** $6: 2'th arg to $4
	** $7: sp of thread to suspend.
	**
	** Totally gross hack: The MIPS calling convention reserves
	** 4 words on the stack for a0..a3.  This routine "ought" to
	** allocate space for callee-save registers plus 4 words for
	** the helper function, but instead we use the 4 words
	** provided by the function that called us (we don't need to
	** save our argument registers).  So what *appears* to be
	** allocating only 40 bytes is actually allocating 56, by
	** using the caller's 16 bytes.
	**
	** The helper routine returns a value that is passed on as the
	** return value from the blocking routine.  Since we don't
	** touch $2 between the helper's return and the end of
	** function, we get this behavior for free.
	*/
        .ent qt_blocki
qt_blocki:
	daddiu $sp,$sp,-80		/* Allocate reg save space. */
	sd $16, 0($sp)
	sd $17, 8($sp)
	sd $18,16($sp)
	sd $19,24($sp)
	sd $20,32($sp)
	sd $21,40($sp)
	sd $22,48($sp)
	sd $23,56($sp)
	sd $30,64($sp)
	sd $31,72($sp)
	daddu $2, $sp,$0		/* $2 <= old sp to pass to func@$4. */
	daddu $sp, $7,$0		/* $sp <= new sp. */
	daddu $25, $4,$0		/* Set helper function proc value. */
	daddu $4, $2,$0		/* $a0 <= pass old sp as a parameter. */
	jalr $25		/* Call helper func@$4 . */
        nop
	ld $31,72($sp)	/* Restore callee-save regs... */
	ld $30,64($sp)
	ld $23,56($sp)
	ld $22,48($sp)
	ld $21,40($sp)
	ld $20,32($sp)
	ld $19,24($sp)
	ld $18,16($sp)
	ld $17, 8($sp)
	ld $16, 0($sp)	/* Restore callee-save */

	daddiu $sp,$sp,80		/* Deallocate reg save space. */
	jr $31			/* Return to caller. */
        nop
        .end qt_blocki

        .ent qt_abort
qt_abort:
	daddu $sp, $7,$0		/* $sp <= new sp. */
	daddu $25, $4,$0		/* Set helper function proc value. */
	daddu $4, $2,$0		/* $a0 <= pass old sp as a parameter. */
	jalr $25		/* Call helper func@$4 . */
        nop
	ld $31,72($sp)	/* Restore callee-save regs... */
	ld $30,64($sp)
	ld $23,56($sp)
	ld $22,48($sp)
	ld $21,40($sp)
	ld $20,32($sp)
	ld $19,24($sp)
	ld $18,16($sp)
	ld $17, 8($sp)
	ld $16, 0($sp)	/* Restore callee-save */

	daddiu $sp,$sp,80		/* Deallocate reg save space. */
	jr $31			/* Return to caller. */
        nop
        .end qt_abort

	/*
	** Non-varargs thread startup.
	** Note: originally, 56 bytes were allocated on the stack.
	** The thread restore routine (_blocki/_abort) removed 40
	** of them, which means there is still 16 bytes for the
	** argument area required by the MIPS calling convention.
	*/
        .ent qt_start
qt_start:
	daddu $4, $16,$0		/* Load up user function pu. */
	daddu $5, $17,$0		/* ... user function pt. */
	daddu $6, $18,$0		/* ... user function userf. */
	daddu $25, $19,$0		/* Set `only' procedure value. */
	jalr $25		/* Call `only'. */
        nop
	dla $25,qt_error		/* Set `qt_error' procedure value. */
	j $25
        nop
        .end qt_start


	/*
	** Save calle-save floating-point regs $f20-$f30
	** See comment in `qt_block' about calling conventinos and
	** reserved space.  Use the same trick here, but here we
	** actually have to allocate all the bytes since we have to
	** leave 4 words leftover for `qt_blocki'.
	**
	** Return value from `qt_block' is the same as the return from
	** `qt_blocki'.  We get that for free since we don't touch $2
	** between the return from `qt_blocki' and the return from
	** `qt_block'.
	*/
        .ent qt_block
qt_block:
	daddiu $sp, $sp,-80		/* 6 8-byte regs, saved ret pc, aligned. */
	sdc1 $f24,  0($sp)
	sdc1 $f25,  8($sp)
	sdc1 $f26, 16($sp)
	sdc1 $f27, 24($sp)
	sdc1 $f28, 32($sp)
	sdc1 $f29, 40($sp)
	sdc1 $f30, 48($sp)
	sdc1 $f31, 56($sp)
	sd $31, 64($sp)
	jal qt_blocki
        nop
	ldc1 $f24,  0($sp)
	ldc1 $f25,  8($sp)
	ldc1 $f26, 16($sp)
	ldc1 $f27, 24($sp)
	ldc1 $f28, 32($sp)
	ldc1 $f29, 40($sp)
	ldc1 $f30, 48($sp)
	ldc1 $f31, 56($sp)
	ld $31, 64($sp)
	daddiu $sp, $sp,80
	j $31
        nop
        .end qt_block


	/*
	** First, call `startup' with the `pt' argument.
	**
	** Next, call the user's function with all arguments.
	** Note that we don't know whether args were passed in
	** integer regs, fp regs, or on the stack (See Gerry Kane
	** "MIPS R2000 RISC Architecture" pg D-22), so we reload
	** all the registers, possibly with garbage arguments.
	**
	** Finally, call `cleanup' with the `pt' argument and with
	** the return value from the user's function.  It is an error
	** for `cleanup' to return.
	*/
        .ent qt_vstart
qt_vstart:
	daddu $4, $17,$0		/* `pt' is arg0 to `startup'. */
	daddu $25, $18,$0		/* Set `startup' procedure value. */
	jal $31, $25		/* Call `startup'. */
        nop

	ld $4,  0($sp)		/* Load up args. */
	ld $5,  8($sp)
	ld $6,  16($sp)
	ld $7, 24($sp)
	ld $8, 32($sp)
	ld $9, 40($sp)
	ld $10, 48($sp)
	ld $11, 56($sp)
	lwc1 $f12, 0($sp)	/* Load up fp args. */
	lwc1 $f14, 8($sp)
	daddu $25, $19,$0		/* Set `userf' procedure value. */
        daddiu $sp, $sp, 64
	jal $31,$25		/* Call `userf'. */
        nop

	daddu $4, $17,$0		/* `pt' is arg0 to `cleanup'. */
	daddu $5, $2,$0		/* Ret. val is arg1 to `cleanup'. */
	daddu $25, $16,$0		/* Set `cleanup' procedure value. */
	jal $31, $25		/* Call `cleanup'. */
        nop

	dla $25,qt_error		/* Set `qt_error' procedure value. */
	j $25
        nop
        .end qt_vstart

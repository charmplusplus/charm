; /*
;  * QuickThreads -- Threads-building toolkit.
;  * Copyright (c) 1993 by David Keppel
;  *
;  * Permission to use, copy, modify and distribute this software and
;  * its documentation for any purpose and without fee is hereby
;  * granted, provided that the above copyright notice and this notice
;  * appear in all copies.  This software is provided as a
;  * proof-of-concept and for demonstration purposes; there is no
;  * representation about the suitability of this software for any
;  * purpose.
;  */

; /* t3e.s -- assembly support. */

	.ident qt$s

; 	/*
; 	** $16: ptr to function to call once curr is suspended
; 	**	and control is on r19's stack.
; 	** $17: 1'th arg to (*$16)(...).
; 	** $18: 2'th arg to (*$16)(...).
; 	** $19: sp of thread to resume.
; 	**
; 	** The helper routine returns a value that is passed on as the
; 	** return value from the blocking routine.  Since we don't
; 	** touch r0 between the helper's return and the end of
; 	** function, we get this behavior for free.
; 	*/

	.extern qt_error
	
	.stack 256
	.psect qt_blocki@code,code,cache
	  
qt_blocki::
	subq r30,80, r30
	stq r26, 0(r30)	
	stq  r9, 8(r30)
	stq r10,16(r30)
	stq r11,24(r30)
	stq r12,32(r30)
	stq r13,40(r30)
	stq r14,48(r30)
	stq r15,56(r30)
	stq r29,64(r30)
	
qt_abort::
	addq r16,r31, r27
	addq r30,r31, r16
	addq r19,r31, r30
	jsr r26,(r27),0

	ldq r26, 0(r30)
	ldq  r9, 8(r30)
	ldq r10,16(r30)
	ldq r11,24(r30)
	ldq r12,32(r30)
	ldq r13,40(r30)
	ldq r14,48(r30)
	ldq r15,56(r30)
	ldq r29,64(r30)

	addq r30,80, r30
	ret r31,(r26),1	
	.endp


; 	/*
; 	** Non-varargs thread startup.
; 	*/
	.stack 256
	.psect qt_start@code,code,cache
qt_start::	
	addq r9,r31,  r16
	addq r10,r31, r17
	addq r11,r31, r18
	addq r12,r31, r27
	jsr r26,(r27),0		

	bsr r26,qt_error	


qt_vstart::	

	addq r9,r31, r16	
	addq r12,r31, r27	
	jsr r26,(r27),0		

	ldt f16, 0(r30)
	ldt f17, 8(r30)
	ldt f18,16(r30)
	ldt f19,24(r30)
	ldt f20,32(r30)
	ldt f21,40(r30)
	ldq r16,48(r30)	
	ldq r17,56(r30)
	ldq r18,64(r30)
	ldq r19,72(r30)
	ldq r20,80(r30)
	ldq r21,88(r30)
	addq r30,96,r30
	addq r11,r31, r27
	jsr r26,(r27),0		

	addq r9,r31, r16	
	addq r0,r31, r17	
	addq r10,r31, r27	
	jsr r26,(r27),0		

	bsr r26,qt_error
	.endp


; 	/*
; 	** Save calle-save floating-point regs f2..f9.
; 	** Also save return pc from whomever called us.
; 	**
; 	** Return value from `qt_block' is the same as the return from
; 	** `qt_blocki'.  We get that for free since we don't touch r0
; 	** between the return from `qt_blocki' and the return from
; 	** `qt_block'.
; 	*/
	.stack 256
	.psect qt_block@code,code,cache
qt_block::	
	subq r30,80, r30
	stq r26, 0(r30)	
	stt f2, 8(r30)
	stt f3,16(r30)
	stt f4,24(r30)
	stt f5,32(r30)
	stt f6,40(r30)
	stt f7,48(r30)
	stt f8,56(r30)
	stt f9,64(r30)

	bsr r26,qt_blocki
				

	ldq r26, 0(r30)		
	ldt f2, 8(r30)
	ldt f3,16(r30)
	ldt f4,24(r30)
	ldt f5,32(r30)
	ldt f6,40(r30)
	ldt f7,48(r30)
	ldt f8,56(r30)
	ldt f9,64(r30)

	addq r30,80, r30	
	ret r31,(r26),1		
	.endp
	.end qt$s

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
	.extern qt_error
	.stack 256
	.psect qt_block@code,code,cache
	  
qt_block::	
qt_blocki::
	subq r30,160, r30
	stt  f2, 24(r30)
	stt  f3, 32(r30)
	stt  f4, 40(r30)
	stt  f5, 48(r30)
	stt  f6, 56(r30)
	stt  f7, 64(r30)
	stt  f8, 72(r30)
	stt  f9, 80(r30)
	stq r26, 88(r30)	
	stq  r9, 96(r30)
	stq r10,104(r30)
	stq r11,112(r30)
	stq r12,120(r30)
	stq r13,128(r30)
	stq r14,136(r30)
	stq r15,144(r30)
	stq r29,152(r30)
qt_abort::
	addq r16,r31, r27
	addq r30,r31, r16
	addq r19,r31, r30
	jsr r26,(r27),0
	ldt  f2, 24(r30)
	ldt  f3, 32(r30)
	ldt  f4, 40(r30)
	ldt  f5, 48(r30)
	ldt  f6, 56(r30)
	ldt  f7, 64(r30)
	ldt  f8, 72(r30)
	ldt  f9, 80(r30)
	ldq r26, 88(r30)
	ldq  r9, 96(r30)
	ldq r10,104(r30)
	ldq r11,112(r30)
	ldq r12,120(r30)
	ldq r13,128(r30)
	ldq r14,136(r30)
	ldq r15,144(r30)
	ldq r29,152(r30)
	addq r30,160, r30
	ret r31,(r26),1	

qt_start::	
	addq r9,r31,  r16
	addq r10,r31, r17
	addq r11,r31, r18
	addq r12,r31, r27
	jsr r26,(r27),0		
	bsr r26,qt_error	

qt_vstart::	
	bsr r26,qt_error

	
	.endp
	.end qt$s


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

        .set noreorder
	.globl b_call_reg
	.globl b_call_imm
	.globl b_add
	.globl b_load

	.ent b_null
b_null:
	j $31
        nop
	.end b_null

	.ent b_call_reg
b_call_reg:
	dla $5,b_null
	daddu $6, $31,0
$L0:
	jal $5
        nop
	jal $5
        nop
	jal $5
        nop
	jal $5
        nop
	jal $5
        nop

	daddu $4, $4,-5
	bgtz $4,$L0
        nop
	j $6
        nop
	.end


	.ent b_call_imm
b_call_imm:
	daddu $6, $31,0
$L1:
	jal b_null
        nop
	jal b_null
        nop
	jal b_null
        nop
	jal b_null
        nop
	jal b_null
        nop

	daddu $4, $4,-5
	bgtz $4,$L1
        nop
	j $6
        nop
	.end


	.ent b_add
b_add:
	daddu $5, $0,$4
	daddu $6, $0,$4
	daddu $7, $0,$4
	daddu $8, $0,$4
$L2:
	daddu $4, $4,-5
	daddu $5, $5,-5
	daddu $6, $6,-5
	daddu $7, $7,-5
	daddu $8, $8,-5

	daddu $4, $4,-5
	daddu $5, $5,-5
	daddu $6, $6,-5
	daddu $7, $7,-5
	daddu $8, $8,-5

	bgtz $4,$L2
        nop
	j $31
        nop
	.end


	.ent b_load
b_load:
$L3:
	ld $0, 0($sp)
	ld $0, 8($sp)
	ld $0, 16($sp)
	ld $0, 24($sp)
	ld $0, 32($sp)

	ld $0, 40($sp)
	ld $0, 48($sp)
	ld $0, 56($sp)
	ld $0, 64($sp)
	ld $0, 72($sp)

	daddu $4, $4,-10
	bgtz $4,$L3
        nop
	j $31
        nop
	.end

/*
  This file exists because the Intel C++ 6.0 Compiler 
for Linux has a bug whereby C++ code that includes 
<iostream> and calls a C function named "init" 
crashes horribly.

  Hence we have to call "init" from here, which is C
and includes no headers.

Orion Sky Lawlor, olawlor@acm.org, 7/26/2002
*/
void init(void);
void fem_impl_call_init(void) {
	init();
}

#ifndef MEGATEST_H
#define MEGATEST_H

#include "charm++.h"

/**
  Start running this test.  
  This routine is called from processor 0 (from a mainchare).
  If the test is reentrant, this function may be called
  multiple times.
  The test finishes successfully with "megatest_finish".
*/
typedef void (*megatest_init_fn)(void);

/// Indicate to megatest that this test finished successfully.
void megatest_finish(void);


/// Do any one-time initialization for this test, like in a main routine.
typedef void (*megatest_moduleinit_fn)(void);
/// Perform Charm++ registration for this test.
typedef void (*megatest_register_fn)(void);

/// Register this new test with megatest.  Must be called 
///  before the mainchare executes; normally via MEGATEST_REGISTER_TEST.
void megatest_register(const char *moduleName,const char *author,
	megatest_init_fn init, megatest_moduleinit_fn moduleinit,
	int reentrant, megatest_register_fn regfn);

/** 
  Use this macro to register your test with megatest.
  
  Place this macro at the bottom of your .C file--
  it uses a global variable class constructor to register
  itself with megatest.

  You must provide:
    - A Charm module called mymod
    - A one-time (mainchare) initialization routine called mymod_moduleinit
    - A per-test initialization routine called mymod_init
*/
#define MEGATEST_REGISTER_TEST(mymod,author,reentrant) \
class mymod##_init_class_t {\
public:\
	mymod##_init_class_t() {\
		megatest_register(#mymod,author,\
			mymod##_init,mymod##_moduleinit,\
			reentrant, _register##mymod); \
	}\
};\
mymod##_init_class_t mymod##_init_class;\


#endif

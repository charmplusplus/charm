// Weak replacement for the *rand48 family of functions which are unavailable
// on non-Cygwin Windows builds. For use in examples only.

#if !defined(RAND48_REPLACEMENT_H)
#define RAND48_REPLACEMENT_H

#if defined(_WIN32) && !defined(__CYGWIN__)

#include <stdlib.h>
#define srand48(x) srand(x)
#define drand48() (((double)rand())/RAND_MAX)

#endif

#endif

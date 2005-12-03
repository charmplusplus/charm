#include "charm-api.h"

#if CMK_FORTRAN_USES_NOSCORE    
/*Fortran and C fallbacks have the same name-- skip this.*/
#else
FDECL void FTN_NAME(INIT,init)(void) { }
#endif

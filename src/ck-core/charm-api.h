/*
Define symbols used to define entry points for API routines
for clients written in C or Fortran.  Useful for building
libraries written in Charm for other languages.
*/
#ifndef __CHARM_API_H
#define __CHARM_API_H

#include "conv-mach.h"
#include "conv-autoconfig.h"

#ifdef __cplusplus
#  define CLINKAGE extern "C"
#else
#  define CLINKAGE /*empty*/
#endif

/*Used to define a "C" entry point*/
#undef CDECL /*<- used by Microsoft Visual C++*/
#define CDECL CLINKAGE

/*Used to define a Fortran-callable routine*/
#define FDECL CLINKAGE

/*Fortran name mangling:
*/
#if CMK_FORTRAN_USES_ALLCAPS
# define FTN_NAME(caps,nocaps) caps  /*Declare name in all caps*/
#elif CMK_FORTRAN_USES_TWOSCORE
# define FTN_NAME(caps,nocaps) nocaps##__ /*No caps, two underscores*/
#elif CMK_FORTRAN_USES_ONESCORE
# define FTN_NAME(caps,nocaps) nocaps##_ /*No caps, one underscore*/
#elif CMK_FORTRAN_USES_NOSCORE
# define FTN_NAME(caps,nocaps) nocaps /*No caps, no underscore*/
#else
/* # error "Did not set fortran name mangling scheme" */
# define FTN_NAME(caps,nocaps) NONMANGLED_##nocaps
#endif

#endif /*Def(thisHeader) */


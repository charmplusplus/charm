/*
Define symbols used to define entry points for API routines
for clients written in C or Fortran.  Useful for building
libraries written in Charm for other languages.
*/
#ifndef __CHARM_API_H
#define __CHARM_API_H

#include "conv-mach.h"

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
#else
# if CMK_FORTRAN_USES_x__
#  define FTN_NAME(caps,nocaps) nocaps##__ /*No caps, two underscores*/
# else
#  define FTN_NAME(caps,nocaps) nocaps##_ /*No caps, one underscore*/
# endif /*__*/
#endif /*ALLCAPS*/

#endif /*Def(thisHeader) */


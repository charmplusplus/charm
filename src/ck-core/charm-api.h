/*
Define symbols used to define entry points for API routines
for clients written in C or Fortran.  Useful for building
libraries written in Charm for other languages.
*/
#ifndef __CHARM_API_H
#define __CHARM_API_H

#include "charm-version.h"
#include "charm-version-git.h"

#include "conv-config.h" /* for CMK_FORTRAN symbols */

/** Used to define a C language entry point*/
#ifdef __cplusplus
#  define CLINKAGE extern "C"
#else
#  define CLINKAGE /*empty*/
#endif

/** Used to define a Fortran-callable routine*/
#define FLINKAGE CLINKAGE

/* Deprecated linkage aliases */
#undef CDECL /*<- used by Microsoft Visual C++*/
#define CDECL CLINKAGE
#define FDECL FLINKAGE

/** Fortran name mangling:
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


/** 
 * Define a new Fortran-callable API routine, returning void,
 * that does nothing but map its arguments onto some 
 * (presumably analogous) C routine. 
 *
 *  @param CAPITALNAME Fortran routine name to define, in all capital letters.
 *  @param Cname C routine to call, in normal case.
 *  @param lowername Fortran routine name in all lowercase.
 *  @param routine_args Fortran routine's arguments (e.g., "int *idx")
 *  @param c_args Arguments to pass to C routine (e.g., "*idx-1")
 */
#define FORTRAN_AS_C(CAPITALNAME,Cname,lowername, routine_args,c_args) \
FLINKAGE void \
FTN_NAME(CAPITALNAME,lowername) routine_args { \
	Cname c_args;\
}

/** 
 * Like FORTRAN_AS_C, but with a return type as the first parameter.
 */
#define FORTRAN_AS_C_RETURN(returnType, CAPITALNAME,Cname,lowername, routine_args,c_args) \
FLINKAGE returnType \
FTN_NAME(CAPITALNAME,lowername) routine_args { \
	return Cname c_args;\
}

/** Shared object symbol exporting */
#ifdef _WIN32
# define CMI_EXPORT __declspec(dllexport)
#elif CMK_HAS_ATTRIBUTE_VISIBILITY_DEFAULT
# define CMI_EXPORT __attribute__((visibility("default")))
#else
# define CMI_EXPORT
#endif

#endif /*Def(thisHeader) */

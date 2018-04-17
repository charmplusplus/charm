/* -*- Mode: C; c-basic-offset:4 ; -*- */
/* 
 *   $Id$    
 *
 *   Copyright (C) 1997 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */

#include "mpioimpl.h"

#ifdef HAVE_WEAK_SYMBOLS

#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_File_c2f = PMPI_File_c2f
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_File_c2f MPI_File_c2f
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_File_c2f as PMPI_File_c2f
/* end of weak pragmas */
#endif

/* Include mapping from MPI->PMPI */
#define MPIO_BUILD_PROFILING
#include "mpioprof.h"
#endif
#include "adio_extern.h"

/*@
    MPI_File_c2f - Translates a C file handle to a Fortran file handle

Input Parameters:
. fh - C file handle (handle)

Return Value:
  Fortran file handle (integer)
@*/
MPI_Fint MPI_File_c2f(MPI_File fh)
{
#ifndef INT_LT_POINTER
    return (MPI_Fint)(intptr_t)fh;
#else
    int i;

    if ((fh <= (MPI_File) 0) || (fh->cookie != ADIOI_FILE_COOKIE))
	return (MPI_Fint) 0;
    if (!CtvAccess(ADIOI_Ftable)) {
	CtvAccess(ADIOI_Ftable_max) = 1024;
	CtvAccess(ADIOI_Ftable) = (MPI_File *)
	    ADIOI_Malloc(CtvAccess(ADIOI_Ftable_max)*sizeof(MPI_File)); 
        CtvAccess(ADIOI_Ftable_ptr) = 0;  /* 0 can't be used though, because 
                                  MPI_FILE_NULL=0 */
	for (i=0; i<CtvAccess(ADIOI_Ftable_max); i++) CtvAccess(ADIOI_Ftable)[i] = MPI_FILE_NULL;
    }
    if (CtvAccess(ADIOI_Ftable_ptr) == CtvAccess(ADIOI_Ftable_max)-1) {
	CtvAccess(ADIOI_Ftable) = (MPI_File *) ADIOI_Realloc(CtvAccess(ADIOI_Ftable), 
                           (CtvAccess(ADIOI_Ftable_max)+1024)*sizeof(MPI_File));
	for (i=CtvAccess(ADIOI_Ftable_max); i<CtvAccess(ADIOI_Ftable_max)+1024; i++) 
	    CtvAccess(ADIOI_Ftable)[i] = MPI_FILE_NULL;
	CtvAccess(ADIOI_Ftable_max) += 1024;
    }
    CtvAccess(ADIOI_Ftable_ptr)++;
    CtvAccess(ADIOI_Ftable)[CtvAccess(ADIOI_Ftable_ptr)] = fh;
    return (MPI_Fint) CtvAccess(ADIOI_Ftable_ptr);
#endif
}

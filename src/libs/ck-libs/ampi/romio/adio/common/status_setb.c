/* -*- Mode: C; c-basic-offset:4 ; -*- */
/* 
 *   $Id$    
 *
 *   Copyright (C) 1997 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */

#include "adio.h"
#include "mpi.h"

int MPIR_Status_set_bytes(MPI_Status *status, MPI_Datatype datatype,
                          int nbytes)
{
    if (status != MPI_STATUS_IGNORE)
        status->MPI_LENGTH=nbytes;
    return MPI_SUCCESS;
}

#if 0
#if defined(MPICH2)
/* Not quite correct, but much closer for MPI2 */
int MPIR_Status_set_bytes(MPI_Status *status, MPI_Datatype datatype, 
			  int nbytes)
{
    if (status != MPI_STATUS_IGNORE)
        MPI_Status_set_elements(status, MPI_BYTE, nbytes);
    return MPI_SUCCESS;
}
#elif defined(MPICH)

void MPID_Status_set_bytes(MPI_Status *status, int nbytes);
int MPIR_Status_set_bytes(MPI_Status *status, MPI_Datatype datatype, 
			  int nbytes);

int MPIR_Status_set_bytes(MPI_Status *status, MPI_Datatype datatype, 
			  int nbytes)
{
    if (status != MPI_STATUS_IGNORE)
        MPID_Status_set_bytes(status, nbytes);
    return MPI_SUCCESS;
}

#endif
#endif

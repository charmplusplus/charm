/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/* 
 *
 *   Copyright (C) 1997 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */

#include "converse.h"

CtvExtern(ADIOI_Flatlist_node*, ADIOI_Flatlist);

CtvExtern(ADIOI_Datarep *, ADIOI_Datarep_head);

/* for f2c and c2f conversion */
CtvExtern(ADIO_File *, ADIOI_Ftable);
CtvExtern(int, ADIOI_Ftable_ptr);
CtvExtern(int, ADIOI_Ftable_max);
CtvExtern(ADIO_Request *, ADIOI_Reqtable);
CtvExtern(int, ADIOI_Reqtable_ptr);
CtvExtern(int, ADIOI_Reqtable_max);
#ifndef HAVE_MPI_INFO
extern MPI_Info *MPIR_Infotable;
extern int MPIR_Infotable_ptr, MPIR_Infotable_max;
#endif
#if defined(ROMIO_XFS) || defined(ROMIO_LUSTRE)
extern int ADIOI_Direct_read, ADIOI_Direct_write;
#endif

CtvExtern(MPI_Errhandler, ADIOI_DFLT_ERR_HANDLER);

CtvExtern(MPI_Info, ADIOI_syshints);

CtvExtern(MPI_Op, ADIO_same_amode);

CtvExtern(int, ADIOI_cb_config_list_keyval);

/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/* 
 *
 *   Copyright (C) 1997 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */

#include "mpio_globals.h"

#define ADIOI_Flatlist (ADIO_Globals()->ADIOI_Flatlist)

#define ADIOI_Datarep_head (ADIO_Globals()->ADIOI_Datarep_head)

/* for f2c and c2f conversion */
#define ADIOI_Ftable (ADIO_Globals()->ADIOI_Ftable)
#define ADIOI_Ftable_ptr (ADIO_Globals()->ADIOI_Ftable_ptr)
#define ADIOI_Ftable_max (ADIO_Globals()->ADIOI_Ftable_max)
#define ADIOI_Reqtable (ADIO_Globals()->ADIOI_Reqtable)
#define ADIOI_Reqtable_ptr (ADIO_Globals()->ADIOI_Reqtable_ptr)
#define ADIOI_Reqtable_max (ADIO_Globals()->ADIOI_Reqtable_max)
#ifndef HAVE_MPI_INFO
#define MPIR_Infotable (ADIO_Globals()->MPIR_Infotable)
#define MPIR_Infotable_ptr (ADIO_Globals()->MPIR_Infotable_ptr)
#define MPIR_Infotable_max (ADIO_Globals()->MPIR_Infotable_max)
#endif
#if defined(ROMIO_XFS) || defined(ROMIO_LUSTRE) || 1
#define ADIOI_Direct_read (ADIO_Globals()->ADIOI_Direct_read)
#define ADIOI_Direct_write (ADIO_Globals()->ADIOI_Direct_write)
#endif

#define ADIO_Init_keyval (ADIO_Globals()->ADIO_Init_keyval)

#define ADIOI_DFLT_ERR_HANDLER (ADIO_Globals()->ADIOI_DFLT_ERR_HANDLER)

#define ADIOI_syshints (ADIO_Globals()->ADIOI_syshints)

#define ADIO_same_amode (ADIO_Globals()->ADIO_same_amode)

#define ADIOI_cb_config_list_keyval (ADIO_Globals()->ADIOI_cb_config_list_keyval)

#define yylval (ADIO_Globals()->yylval)
#define token_ptr (ADIO_Globals()->token_ptr)

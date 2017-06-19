/* -*- Mode: C; c-basic-offset:4 ; -*- */
/* 
 *   $Id$    
 *
 *   Copyright (C) 1997 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */

#include "converse.h" //For Ctv*

CtvExtern(ADIOI_Flatlist_node*, ADIOI_Flatlist);

CtvExtern(ADIOI_Async_node*, ADIOI_Async_list_head);
CtvExtern(ADIOI_Async_node*, ADIOI_Async_list_tail);

CtvExtern(ADIOI_Async_node*, ADIOI_Async_avail_head);
CtvExtern(ADIOI_Async_node*, ADIOI_Async_avail_tail);

CtvExtern(ADIOI_Malloc_async*, ADIOI_Malloc_async_head);
CtvExtern(ADIOI_Malloc_async*, ADIOI_Malloc_async_tail);

CtvExtern(ADIOI_Req_node*, ADIOI_Req_avail_head);
CtvExtern(ADIOI_Req_node*, ADIOI_Req_avail_tail);

CtvExtern(ADIOI_Malloc_req*, ADIOI_Malloc_req_head);
CtvExtern(ADIOI_Malloc_req*, ADIOI_Malloc_req_tail);
// extern ADIOI_Async_node *ADIOI_Async_list_head, *ADIOI_Async_list_tail; 
// /* list of outstanding asynchronous requests */

// extern ADIOI_Async_node *ADIOI_Async_avail_head, *ADIOI_Async_avail_tail;
// /* list of available (already malloced) nodes for the async list */
// extern ADIOI_Malloc_async *ADIOI_Malloc_async_head, *ADIOI_Malloc_async_tail;
// /* list of malloced areas in memory, which must be freed in ADIO_End */

// extern ADIOI_Req_node *ADIOI_Req_avail_head, *ADIOI_Req_avail_tail;
//     /* list of available (already malloced) request objects */
// extern ADIOI_Malloc_req *ADIOI_Malloc_req_head, *ADIOI_Malloc_req_tail;
//     /* list of malloced areas for requests, which must be freed in ADIO_End */

/* for f2c and c2f conversion */
// extern ADIO_File *ADIOI_Ftable;
// extern int ADIOI_Ftable_ptr, ADIOI_Ftable_max;
// extern ADIO_Request *ADIOI_Reqtable;
// extern int ADIOI_Reqtable_ptr, ADIOI_Reqtable_max;
CtvExtern(ADIO_File*, ADIOI_Ftable);
CtvExtern(int, ADIOI_Ftable_ptr);
CtvExtern(int, ADIOI_Ftable_max);
CtvExtern(ADIO_Request*, ADIOI_Reqtable);
CtvExtern(int, ADIOI_Reqtable_ptr);
CtvExtern(int, ADIOI_Reqtable_max);
#ifndef HAVE_MPI_INFO
extern MPI_Info *MPIR_Infotable;
extern int MPIR_Infotable_ptr, MPIR_Infotable_max;
#endif
#ifdef XFS
extern int ADIOI_Direct_read, ADIOI_Direct_write;
#endif

CtvExtern(MPI_Errhandler, ADIOI_DFLT_ERR_HANDLER);

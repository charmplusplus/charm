/* -*- Mode: C; c-basic-offset:4 ; -*- */
/* 
 *   $Id$    
 *
 *   Copyright (C) 1997 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */

#include "adio.h"
#include "converse.h" //For Ctv*

CtvDeclare(ADIOI_Flatlist_node*, ADIOI_Flatlist);

CtvDeclare(ADIOI_Async_node*, ADIOI_Async_list_head);
CtvDeclare(ADIOI_Async_node*, ADIOI_Async_list_tail);

CtvDeclare(ADIOI_Async_node*, ADIOI_Async_avail_head);
CtvDeclare(ADIOI_Async_node*, ADIOI_Async_avail_tail);

CtvDeclare(ADIOI_Malloc_async*, ADIOI_Malloc_async_head);
CtvDeclare(ADIOI_Malloc_async*, ADIOI_Malloc_async_tail);
// ADIOI_Async_node *ADIOI_Async_list_head, *ADIOI_Async_list_tail;
    /* list of outstanding asynchronous requests */
// ADIOI_Async_node *ADIOI_Async_avail_head, *ADIOI_Async_avail_tail;
    /* list of available (already malloced) nodes for above async list */
// ADIOI_Malloc_async *ADIOI_Malloc_async_head, *ADIOI_Malloc_async_tail;
  /* list of malloced areas for async_list, which must be freed in ADIO_End */

CtvDeclare(ADIOI_Req_node*, ADIOI_Req_avail_head);
CtvDeclare(ADIOI_Req_node*, ADIOI_Req_avail_tail);

CtvDeclare(ADIOI_Malloc_req*, ADIOI_Malloc_req_head);
CtvDeclare(ADIOI_Malloc_req*, ADIOI_Malloc_req_tail);
// ADIOI_Req_node *ADIOI_Req_avail_head, *ADIOI_Req_avail_tail;
    /* list of available (already malloced) request objects */
// ADIOI_Malloc_req *ADIOI_Malloc_req_head, *ADIOI_Malloc_req_tail;
    /* list of malloced areas for requests, which must be freed in ADIO_End */

/* for f2c and c2f conversion */
CtvDeclare(ADIO_File*, ADIOI_Ftable);
CtvDeclare(int, ADIOI_Ftable_ptr);
CtvDeclare(int, ADIOI_Ftable_max);
CtvDeclare(ADIO_Request*, ADIOI_Reqtable);
CtvDeclare(int, ADIOI_Reqtable_ptr);
CtvDeclare(int, ADIOI_Reqtable_max);
#ifndef HAVE_MPI_INFO
MPI_Info *MPIR_Infotable;
int MPIR_Infotable_ptr, MPIR_Infotable_max;
#endif

#ifdef XFS
int ADIOI_Direct_read, ADIOI_Direct_write;
#endif

CtvDeclare(int, ADIO_Init_keyval); //=MPI_KEYVAL_INVALID;
CtvDeclare(int, ADIO_Init_keyval_done); // accessed in open.c, delete.c

CtvDeclare(MPI_Errhandler, ADIOI_DFLT_ERR_HANDLER); // = MPI_ERRORS_RETURN;

void ADIO_Init(int *argc, char ***argv, int *error_code)
{
#ifdef XFS
    char *c;
#endif
    CtvInitialize(ADIOI_Flatlist_node*, ADIOI_Flatlist);

    CtvInitialize(ADIOI_Async_node*, ADIOI_Async_list_head);
    CtvInitialize(ADIOI_Async_node*, ADIOI_Async_list_tail);

    CtvInitialize(ADIOI_Async_node*, ADIOI_Async_avail_head);
    CtvInitialize(ADIOI_Async_node*, ADIOI_Async_avail_tail);

    CtvInitialize(ADIOI_Malloc_async*, ADIOI_Malloc_async_head);
    CtvInitialize(ADIOI_Malloc_async*, ADIOI_Malloc_async_tail);

    CtvInitialize(ADIOI_Req_node*, ADIOI_Req_avail_head);
    CtvInitialize(ADIOI_Req_node*, ADIOI_Req_avail_tail);

    CtvInitialize(ADIOI_Malloc_req*, ADIOI_Malloc_req_head);
    CtvInitialize(ADIOI_Malloc_req*, ADIOI_Malloc_req_tail);

    CtvInitialize(ADIO_File*, ADIOI_Ftable);
    CtvInitialize(int, ADIOI_Ftable_ptr);
    CtvInitialize(int, ADIOI_Ftable_max);
    CtvInitialize(ADIO_Request*, ADIOI_Reqtable);
    CtvInitialize(int, ADIOI_Reqtable_ptr);
    CtvInitialize(int, ADIOI_Reqtable_max);

    CtvInitialize(int, ADIO_Init_keyval);
    // ADIO_Init_keyval is initialized to MPI_KEYVAL_INVALID in open.c/delete.c

    CtvInitialize(int, ADIO_Init_keyval_done);

    CtvInitialize(MPI_Errhandler, ADIOI_DFLT_ERR_HANDLER);
    CtvAccess(ADIOI_DFLT_ERR_HANDLER) = MPI_ERRORS_RETURN;

/* initialize the linked list containing flattened datatypes */
    CtvAccess(ADIOI_Flatlist) = (ADIOI_Flatlist_node *) ADIOI_Malloc(sizeof(ADIOI_Flatlist_node));
    CtvAccess(ADIOI_Flatlist)->type = (MPI_Datatype) NULL;
    CtvAccess(ADIOI_Flatlist)->next = NULL;
    CtvAccess(ADIOI_Flatlist)->blocklens = NULL;
    CtvAccess(ADIOI_Flatlist)->indices = NULL;

    CtvAccess(ADIOI_Async_list_head) = CtvAccess(ADIOI_Async_list_tail) = NULL;
    CtvAccess(ADIOI_Async_avail_head) = CtvAccess(ADIOI_Async_avail_tail) = NULL;
    CtvAccess(ADIOI_Malloc_async_head) = CtvAccess(ADIOI_Malloc_async_tail) = NULL;

    CtvAccess(ADIOI_Req_avail_head) = CtvAccess(ADIOI_Req_avail_tail) = NULL;
    CtvAccess(ADIOI_Malloc_req_head) = CtvAccess(ADIOI_Malloc_req_tail) = NULL;

    CtvAccess(ADIOI_Ftable) = NULL;
    CtvAccess(ADIOI_Ftable_ptr) = CtvAccess(ADIOI_Ftable_max) = 0;

    CtvAccess(ADIOI_Reqtable) = NULL;
    CtvAccess(ADIOI_Reqtable_ptr) = CtvAccess(ADIOI_Reqtable_max) = 0;

#ifndef HAVE_MPI_INFO
    MPIR_Infotable = NULL;
    MPIR_Infotable_ptr = MPIR_Infotable_max = 0;
#endif

#ifdef XFS
    c = getenv("MPIO_DIRECT_READ");
    if (c && (!strcmp(c, "true") || !strcmp(c, "TRUE"))) 
	ADIOI_Direct_read = 1;
    else ADIOI_Direct_read = 0;
    c = getenv("MPIO_DIRECT_WRITE");
    if (c && (!strcmp(c, "true") || !strcmp(c, "TRUE"))) 
	ADIOI_Direct_write = 1;
    else ADIOI_Direct_write = 0;
#endif

    *error_code = MPI_SUCCESS;
}

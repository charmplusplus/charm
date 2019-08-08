/* -*- Mode: C; c-basic-offset:4 ; -*- */
/* 
 *   $Id$    
 *
 *   Copyright (C) 1997 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */
 
 
/*
 * This file is separate from mpio.h because ROMIO is included as part
 * of AMPI's implementation of the MPI standard, and certain global variable
 * privatization methods require the AMPI API to be exposed as function pointers
 * through a shim and loader mechanism that needs to list the entire set of
 * provided functions at multiple points in its implementation.
 *
 * See src/libs/ck-libs/ampi/ampi_functions.h for mandatory procedures.
 *
 * For ease of reading: AMPI_CUSTOM_FUNC(ReturnType, FunctionName, Parameters...)
 */


/* MPI-IO function prototypes */

/* The compiler must support ANSI C style prototypes, otherwise 
   long long constants (e.g. 0) may get passed as ints. */

#ifndef HAVE_PRAGMA_HP_SEC_DEF

/* Section 9.2 */
/* Begin Prototypes */
AMPI_CUSTOM_FUNC(int, MPI_File_open, MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh)
AMPI_CUSTOM_FUNC(int, MPI_File_close, MPI_File *fh)
AMPI_CUSTOM_FUNC(int, MPI_File_delete, const char *filename, MPI_Info info)
AMPI_CUSTOM_FUNC(int, MPI_File_set_size, MPI_File fh, MPI_Offset size)
AMPI_CUSTOM_FUNC(int, MPI_File_preallocate, MPI_File fh, MPI_Offset size)
AMPI_CUSTOM_FUNC(int, MPI_File_get_size, MPI_File fh, MPI_Offset *size)
AMPI_CUSTOM_FUNC(int, MPI_File_get_group, MPI_File fh, MPI_Group *group)
AMPI_CUSTOM_FUNC(int, MPI_File_get_amode, MPI_File fh, int *amode)
AMPI_CUSTOM_FUNC(int, MPI_File_set_info, MPI_File fh, MPI_Info info)
AMPI_CUSTOM_FUNC(int, MPI_File_get_info, MPI_File fh, MPI_Info *info_used)

/* Section 9.3 */
AMPI_CUSTOM_FUNC(int, MPI_File_set_view, MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype,
                      const char *datarep, MPI_Info info)
AMPI_CUSTOM_FUNC(int, MPI_File_get_view, MPI_File fh, MPI_Offset *disp, MPI_Datatype *etype, MPI_Datatype *filetype,
                      char *datarep)

/* Section 9.4.2 */
AMPI_CUSTOM_FUNC(int, MPI_File_read_at, MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype,
                     MPI_Status *status)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, MPI_File_read_at_all, MPI_File fh, MPI_Offset offset, void * buf, int count,
                         MPI_Datatype datatype, MPI_Status *status)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, MPI_File_write_at, MPI_File fh, MPI_Offset offset, const void * buf, int count,
                      MPI_Datatype datatype, MPI_Status *status)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, MPI_File_write_at_all, MPI_File fh, MPI_Offset offset, const void *buf, int count,
                          MPI_Datatype datatype, MPI_Status *status)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)

/* nonblocking calls currently use MPIO_Request, because generalized
   requests not yet implemented. For the same reason, MPIO_Test and 
   MPIO_Wait are used to test and wait on nonblocking I/O requests */ 
AMPI_CUSTOM_FUNC(int, MPI_File_iread_at, MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype,
                      MPIO_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, MPI_File_iwrite_at, MPI_File fh, MPI_Offset offset, const void *buf, int count,
                       MPI_Datatype datatype, MPIO_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)

/* Section 9.4.3 */
AMPI_CUSTOM_FUNC(int, MPI_File_read, MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_read_all, MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Status *status)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_write, MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                   MPI_Status *status)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_write_all, MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                       MPI_Status *status)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)

/* nonblocking calls currently use MPIO_Request, because generalized
   requests not yet implemented. For the same reason, MPIO_Test and 
   MPIO_Wait are used to test and wait on nonblocking I/O requests */ 

AMPI_CUSTOM_FUNC(int, MPI_File_iread, MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPIO_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_iwrite, MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                    MPIO_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)

AMPI_CUSTOM_FUNC(int, MPI_File_seek, MPI_File fh, MPI_Offset offset, int whence)
AMPI_CUSTOM_FUNC(int, MPI_File_get_position, MPI_File fh, MPI_Offset *offset)
AMPI_CUSTOM_FUNC(int, MPI_File_get_byte_offset, MPI_File fh, MPI_Offset offset, MPI_Offset *disp)

/* Section 9.4.4 */
AMPI_CUSTOM_FUNC(int, MPI_File_read_shared, MPI_File fh, void *buf, int count, MPI_Datatype datatype,
                         MPI_Status *status)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_write_shared, MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                          MPI_Status *status)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_iread_shared, MPI_File fh, void *buf, int count, MPI_Datatype datatype,
                          MPIO_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_iwrite_shared, MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                           MPIO_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_read_ordered, MPI_File fh, void *buf, int count, MPI_Datatype datatype,
                          MPI_Status *status)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_write_ordered, MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                           MPI_Status *status)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_seek_shared, MPI_File fh, MPI_Offset offset, int whence)
AMPI_CUSTOM_FUNC(int, MPI_File_get_position_shared, MPI_File fh, MPI_Offset *offset)

/* Section 9.4.5 */
AMPI_CUSTOM_FUNC(int, MPI_File_read_at_all_begin, MPI_File fh, MPI_Offset offset, void *buf, int count,
                               MPI_Datatype datatype)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, MPI_File_read_at_all_end, MPI_File fh, void *buf, MPI_Status *status)
AMPI_CUSTOM_FUNC(int, MPI_File_write_at_all_begin, MPI_File fh, MPI_Offset offset, const void *buf, int count,
                                MPI_Datatype datatype)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, MPI_File_write_at_all_end, MPI_File fh, const void *buf, MPI_Status *status)
AMPI_CUSTOM_FUNC(int, MPI_File_read_all_begin, MPI_File fh, void *buf, int count, MPI_Datatype datatype)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_read_all_end, MPI_File fh, void *buf, MPI_Status *status)
AMPI_CUSTOM_FUNC(int, MPI_File_write_all_begin, MPI_File fh, const void *buf, int count, MPI_Datatype datatype)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_write_all_end, MPI_File fh, const void *buf, MPI_Status *status)
AMPI_CUSTOM_FUNC(int, MPI_File_read_ordered_begin, MPI_File fh, void *buf, int count, MPI_Datatype datatype)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_read_ordered_end, MPI_File fh, void *buf, MPI_Status *status)
AMPI_CUSTOM_FUNC(int, MPI_File_write_ordered_begin, MPI_File fh, const void *buf, int count, MPI_Datatype datatype)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_write_ordered_end, MPI_File fh, const void *buf, MPI_Status *status)

/* Section 9.5.1 */
AMPI_CUSTOM_FUNC(int, MPI_File_get_type_extent, MPI_File fh, MPI_Datatype datatype, MPI_Aint *extent)

/* Section 9.5.3 */
AMPI_CUSTOM_FUNC(int, MPI_Register_datarep, const char *datarep, MPI_Datarep_conversion_function *read_conversion_fn,
			 MPI_Datarep_conversion_function *write_conversion_fn,
			 MPI_Datarep_extent_function *dtype_file_extent_fn, void *extra_state)

/* Section 9.6.1 */
AMPI_CUSTOM_FUNC(int, MPI_File_set_atomicity, MPI_File fh, int flag)
AMPI_CUSTOM_FUNC(int, MPI_File_get_atomicity, MPI_File fh, int *flag)
AMPI_CUSTOM_FUNC(int, MPI_File_sync, MPI_File fh)

/* Section 4.13.3 */
#ifndef MPICH
/* MPICH provides these definitions */
AMPI_CUSTOM_FUNC(int, MPI_File_set_errhandler, MPI_File file, MPI_Errhandler errhandler)
AMPI_CUSTOM_FUNC(int, MPI_File_get_errhandler, MPI_File file, MPI_Errhandler *errhandler)
#endif

/* For MPI 3.1 */
AMPI_CUSTOM_FUNC(int, MPI_File_iread_at_all, MPI_File fh, MPI_Offset offset, void *buf, int count,
                           MPI_Datatype datatype, MPI_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, MPI_File_iwrite_at_all, MPI_File fh, MPI_Offset offset, const void *buf, int count,
                            MPI_Datatype datatype, MPI_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, MPI_File_iread_all, MPI_File fh, void *buf, int count, MPI_Datatype datatype,
                        MPI_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, MPI_File_iwrite_all, MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                         MPI_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
/* End Prototypes */

#ifndef HAVE_MPI_DARRAY_SUBARRAY
/* Section 4.14.4 */
AMPI_CUSTOM_FUNC(int, MPI_Type_create_subarray, int ndims, const int array_of_sizes[], const int array_of_subsizes[],
                             const int array_of_starts[], int order, MPI_Datatype oldtype,
                             MPI_Datatype *newtype)

/* Section 4.14.5 */
AMPI_CUSTOM_FUNC(int, MPI_Type_create_darray, int size, int rank, int ndims, const int array_of_gsizes[],
                           const int array_of_distribs[], const int array_of_dargs[],
                           const int array_of_psizes, int order, MPI_Datatype oldtype,
                           MPI_Datatype *newtype)
#endif

/* The globus2 device has to rename MPI_ symbols in order to use the vendor
   MPI as one of its transport mechanisms.  Therefore, the following undefines
   should only happen if MPICH_RENAMING_MPI_FUNCS is not defined. */
/* Section 4.12.4 */
#if !defined(MPICH_RENAMING_MPI_FUNCS)
#ifdef MPI_File_f2c
#undef MPI_File_f2c
#endif
#ifdef MPI_File_c2f
#undef MPI_File_c2f
#endif
#endif
/* above needed for some versions of mpi.h in MPICH!! */
AMPI_CUSTOM_FUNC(MPI_File, MPI_File_f2c, MPI_Fint file)
AMPI_CUSTOM_FUNC(MPI_Fint, MPI_File_c2f, MPI_File file)


#ifndef HAVE_MPI_GREQUEST
/* The following functions are required if generalized requests are not
   available, because in that case, an MPIO_Request object
   is currently used for nonblocking I/O. */
AMPI_CUSTOM_FUNC(int, MPIO_Test, MPIO_Request *request, int *flag, MPI_Status *status)
AMPI_CUSTOM_FUNC(int, MPIO_Wait, MPIO_Request *request, MPI_Status *status)
AMPI_CUSTOM_FUNC(int, MPIO_Testall, int count, MPIO_Request array_of_requests[], int *flag,
                 MPI_Status array_of_statuses[])
AMPI_CUSTOM_FUNC(int, MPIO_Waitall, int count, MPIO_Request array_of_requests[], MPI_Status array_of_statuses[])
AMPI_CUSTOM_FUNC(int, MPIO_Testany, int count, MPIO_Request array_of_requests[], int *indx, int *flag,
                 MPI_Status *status)
AMPI_CUSTOM_FUNC(int, MPIO_Waitany, int count, MPIO_Request array_of_requests[], int *indx, MPI_Status *status)
AMPI_CUSTOM_FUNC(int, MPIO_Waitsome, int incount, MPIO_Request array_of_requests[], int *outcount,
                  int array_of_indices[], MPI_Status array_of_statuses[])
AMPI_CUSTOM_FUNC(int, MPIO_Testsome, int incount, MPIO_Request array_of_requests[], int *outcount,
                  int array_of_indices[], MPI_Status array_of_statuses[])

AMPI_CUSTOM_FUNC(MPI_Fint, MPIO_Request_c2f, MPIO_Request request)
AMPI_CUSTOM_FUNC(MPIO_Request, MPIO_Request_f2c, MPI_Fint request)
#endif /* HAVE_MPI_GREQUEST */

/* info functions if not defined in the MPI implementation */
#ifndef HAVE_MPI_INFO

AMPI_CUSTOM_FUNC(int, MPI_Info_create, MPI_Info *info)
AMPI_CUSTOM_FUNC(int, MPI_Info_set, MPI_Info info, const char *key, const char *value)
AMPI_CUSTOM_FUNC(int, MPI_Info_delete, MPI_Info info, const char *key)
AMPI_CUSTOM_FUNC(int, MPI_Info_get, MPI_Info info, const char *key, int valuelen, char *value, int *flag)
AMPI_CUSTOM_FUNC(int, MPI_Info_get_valuelen, MPI_Info info, const char *key, int *valuelen, int *flag)
AMPI_CUSTOM_FUNC(int, MPI_Info_get_nkeys, MPI_Info info, int *nkeys)
AMPI_CUSTOM_FUNC(int, MPI_Info_get_nthkey, MPI_Info info, int n, char *key)
AMPI_CUSTOM_FUNC(int, MPI_Info_dup, MPI_Info info, MPI_Info *newinfo)
AMPI_CUSTOM_FUNC(int, MPI_Info_free, MPI_Info *info)

/* The globus2 device has to rename MPI_ symbols in order to use the vendor
   MPI as one of its transport mechanisms.  Therefore, the following undefines
   should only happen if MPICH_RENAMING_MPI_FUNCS is not defined. */
#if !defined(MPICH_RENAMING_MPI_FUNCS)
#ifdef MPI_Info_f2c
#undef MPI_Info_f2c
#endif
#ifdef MPI_Info_c2f
#undef MPI_Info_c2f
#endif
#endif
/* above needed for some versions of mpi.h in MPICH!! */
AMPI_CUSTOM_FUNC(MPI_Fint, MPI_Info_c2f, MPI_Info info)
AMPI_CUSTOM_FUNC(MPI_Info, MPI_Info_f2c, MPI_Fint info)
#endif

#endif   /* HAVE_PRAGMA_HP_SEC_DEF */


/**************** BINDINGS FOR THE PROFILING INTERFACE ***************/


/* Section 9.2 */
AMPI_CUSTOM_FUNC(int, PMPI_File_open, MPI_Comm, const char *, int, MPI_Info, MPI_File *)
AMPI_CUSTOM_FUNC(int, PMPI_File_close, MPI_File *)
AMPI_CUSTOM_FUNC(int, PMPI_File_delete, const char *, MPI_Info)
AMPI_CUSTOM_FUNC(int, PMPI_File_set_size, MPI_File, MPI_Offset)
AMPI_CUSTOM_FUNC(int, PMPI_File_preallocate, MPI_File, MPI_Offset)
AMPI_CUSTOM_FUNC(int, PMPI_File_get_size, MPI_File, MPI_Offset *)
AMPI_CUSTOM_FUNC(int, PMPI_File_get_group, MPI_File, MPI_Group *)
AMPI_CUSTOM_FUNC(int, PMPI_File_get_amode, MPI_File, int *)
AMPI_CUSTOM_FUNC(int, PMPI_File_set_info, MPI_File, MPI_Info)
AMPI_CUSTOM_FUNC(int, PMPI_File_get_info, MPI_File, MPI_Info *)

/* Section 9.3 */
AMPI_CUSTOM_FUNC(int, PMPI_File_set_view, MPI_File, MPI_Offset,
    MPI_Datatype, MPI_Datatype, const char *, MPI_Info)
AMPI_CUSTOM_FUNC(int, PMPI_File_get_view, MPI_File, MPI_Offset *,
      MPI_Datatype *, MPI_Datatype *, char *)

/* Section 9.4.2 */
AMPI_CUSTOM_FUNC(int, PMPI_File_read_at, MPI_File, MPI_Offset, void *,
	      int, MPI_Datatype, MPI_Status *)
              // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, PMPI_File_read_at_all, MPI_File, MPI_Offset, void *,
	      int, MPI_Datatype, MPI_Status *)
              // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, PMPI_File_write_at, MPI_File, MPI_Offset, const void *,
	      int, MPI_Datatype, MPI_Status *)
              // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, PMPI_File_write_at_all, MPI_File, MPI_Offset, const void *,
	      int, MPI_Datatype, MPI_Status *)
              // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)

/* nonblocking calls currently use MPIO_Request, because generalized
   requests not yet implemented. For the same reason, MPIO_Test and 
   MPIO_Wait are used to test and wait on nonblocking I/O requests */ 

AMPI_CUSTOM_FUNC(int, PMPI_File_iread_at, MPI_File, MPI_Offset, void *,
	      int, MPI_Datatype, MPIO_Request *)
              // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, PMPI_File_iwrite_at, MPI_File, MPI_Offset, const void *,
	      int, MPI_Datatype, MPIO_Request *)
              // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)

/* Section 9.4.3 */
AMPI_CUSTOM_FUNC(int, PMPI_File_read, MPI_File, void *, int, MPI_Datatype, MPI_Status *)
                   // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_read_all, MPI_File, void *, int, MPI_Datatype, MPI_Status *)
                       // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_write, MPI_File, const void *, int, MPI_Datatype, MPI_Status *)
                    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_write_all, MPI_File, const void *, int, MPI_Datatype, MPI_Status *)
                        // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)

/* nonblocking calls currently use MPIO_Request, because generalized
   requests not yet implemented. For the same reason, MPIO_Test and 
   MPIO_Wait are used to test and wait on nonblocking I/O requests */ 

AMPI_CUSTOM_FUNC(int, PMPI_File_iread, MPI_File, void *, int, MPI_Datatype, MPIO_Request *)
                    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_iwrite, MPI_File, const void *, int, MPI_Datatype, MPIO_Request *)
                     // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)

AMPI_CUSTOM_FUNC(int, PMPI_File_seek, MPI_File, MPI_Offset, int)
AMPI_CUSTOM_FUNC(int, PMPI_File_get_position, MPI_File, MPI_Offset *)
AMPI_CUSTOM_FUNC(int, PMPI_File_get_byte_offset, MPI_File, MPI_Offset, MPI_Offset *)

/* Section 9.4.4 */
AMPI_CUSTOM_FUNC(int, PMPI_File_read_shared, MPI_File, void *, int, MPI_Datatype, MPI_Status *)
                          // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_write_shared, MPI_File, const void *, int, MPI_Datatype, MPI_Status *)
                           // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_iread_shared, MPI_File, void *, int,
			   MPI_Datatype, MPIO_Request *)
                           // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_iwrite_shared, MPI_File, const void *, int,
			    MPI_Datatype, MPIO_Request *)
                            // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_read_ordered, MPI_File, void *, int, MPI_Datatype, MPI_Status *)
                           // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_write_ordered, MPI_File, const void *, int, MPI_Datatype, MPI_Status *)
                            // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_seek_shared, MPI_File, MPI_Offset, int)
AMPI_CUSTOM_FUNC(int, PMPI_File_get_position_shared, MPI_File, MPI_Offset *)

/* Section 9.4.5 */
AMPI_CUSTOM_FUNC(int, PMPI_File_read_at_all_begin, MPI_File, MPI_Offset, void *,
                               int, MPI_Datatype)
                               // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, PMPI_File_read_at_all_end, MPI_File, void *, MPI_Status *)
AMPI_CUSTOM_FUNC(int, PMPI_File_write_at_all_begin, MPI_File, MPI_Offset, const void *,
                                 int, MPI_Datatype)
                                 // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, PMPI_File_write_at_all_end, MPI_File, const void *, MPI_Status *)
AMPI_CUSTOM_FUNC(int, PMPI_File_read_all_begin, MPI_File, void *, int, MPI_Datatype)
                             // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_read_all_end, MPI_File, void *, MPI_Status *)
AMPI_CUSTOM_FUNC(int, PMPI_File_write_all_begin, MPI_File, const void *, int, MPI_Datatype)
                              // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_write_all_end, MPI_File, const void *, MPI_Status *)
AMPI_CUSTOM_FUNC(int, PMPI_File_read_ordered_begin, MPI_File, void *, int, MPI_Datatype)
                                 // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_read_ordered_end, MPI_File, void *, MPI_Status *)
AMPI_CUSTOM_FUNC(int, PMPI_File_write_ordered_begin, MPI_File, const void *, int, MPI_Datatype)
                                  // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_write_ordered_end, MPI_File, const void *, MPI_Status *)

/* Section 9.5.1 */
AMPI_CUSTOM_FUNC(int, PMPI_File_get_type_extent, MPI_File, MPI_Datatype, MPI_Aint *)

/* Section 9.5.3 */
AMPI_CUSTOM_FUNC(int, PMPI_Register_datarep, const char *,
			 MPI_Datarep_conversion_function *,
			 MPI_Datarep_conversion_function *,
			 MPI_Datarep_extent_function *,
			 void *)

/* Section 9.6.1 */
AMPI_CUSTOM_FUNC(int, PMPI_File_set_atomicity, MPI_File, int)
AMPI_CUSTOM_FUNC(int, PMPI_File_get_atomicity, MPI_File, int *)
AMPI_CUSTOM_FUNC(int, PMPI_File_sync, MPI_File)

/* Section 4.13.3 */
#ifndef MPICH
/* MPICH provides these definitions */
AMPI_CUSTOM_FUNC(int, PMPI_File_set_errhandler,  MPI_File, MPI_Errhandler )
AMPI_CUSTOM_FUNC(int, PMPI_File_get_errhandler,  MPI_File, MPI_Errhandler * )
#endif

/* For MPI 3.1 */
AMPI_CUSTOM_FUNC(int, PMPI_File_iread_at_all, MPI_File fh, MPI_Offset offset, void *buf, int count,
                            MPI_Datatype datatype, MPI_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, PMPI_File_iwrite_at_all, MPI_File fh, MPI_Offset offset, const void *buf, int count,
                             MPI_Datatype datatype, MPI_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(3,5)
AMPI_CUSTOM_FUNC(int, PMPI_File_iread_all, MPI_File fh, void *buf, int count, MPI_Datatype datatype,
                         MPI_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)
AMPI_CUSTOM_FUNC(int, PMPI_File_iwrite_all, MPI_File fh, const void *buf, int count, MPI_Datatype datatype,
                          MPI_Request *request)
    // MPICH_ATTR_POINTER_WITH_TYPE_TAG(2,4)

#ifndef HAVE_MPI_DARRAY_SUBARRAY
/* Section 4.14.4 */
AMPI_CUSTOM_FUNC(int, PMPI_Type_create_subarray, int, int *, int *, int *, int,
                      MPI_Datatype, MPI_Datatype *)

/* Section 4.14.5 */
AMPI_CUSTOM_FUNC(int, PMPI_Type_create_darray, int, int, int, int *, int *,
                    int *, int *, int, MPI_Datatype, MPI_Datatype *)
#endif

/* Section 4.12.4 */
AMPI_CUSTOM_FUNC(MPI_File, PMPI_File_f2c, MPI_Fint)
AMPI_CUSTOM_FUNC(MPI_Fint, PMPI_File_c2f, MPI_File)

#ifndef HAVE_MPI_GREQUEST
/* The following functions are required if generalized requests are not
   available, because in that case, an MPIO_Request object
   is currently used for nonblocking I/O. */
AMPI_CUSTOM_FUNC(int, PMPIO_Test, MPIO_Request *, int *, MPI_Status *)
AMPI_CUSTOM_FUNC(int, PMPIO_Wait, MPIO_Request *, MPI_Status *)
AMPI_CUSTOM_FUNC(int, PMPIO_Testall, int, MPIO_Request *, int *, MPI_Status *)
AMPI_CUSTOM_FUNC(int, PMPIO_Waitall, int, MPIO_Request *, MPI_Status *)
AMPI_CUSTOM_FUNC(int, PMPIO_Testany, int, MPIO_Request *, int *, int *, MPI_Status *)
AMPI_CUSTOM_FUNC(int, PMPIO_Waitany, int, MPIO_Request *, int *, MPI_Status *)
AMPI_CUSTOM_FUNC(int, PMPIO_Waitsome, int, MPIO_Request *, int *, int *, MPI_Status *)
AMPI_CUSTOM_FUNC(int, PMPIO_Testsome, int, MPIO_Request *, int *, int *, MPI_Status *)
AMPI_CUSTOM_FUNC(MPI_Fint, PMPIO_Request_c2f, MPIO_Request)
AMPI_CUSTOM_FUNC(MPIO_Request, PMPIO_Request_f2c, MPI_Fint)
#endif /* HAVE_MPI_GREQUEST */

/* info functions if not defined in the MPI implementation */
#ifndef HAVE_MPI_INFO

AMPI_CUSTOM_FUNC(int, PMPI_Info_create, MPI_Info *)
AMPI_CUSTOM_FUNC(int, PMPI_Info_set, MPI_Info, char *, char *)
AMPI_CUSTOM_FUNC(int, PMPI_Info_delete, MPI_Info, char *)
AMPI_CUSTOM_FUNC(int, PMPI_Info_get, MPI_Info, char *, int, char *, int *)
AMPI_CUSTOM_FUNC(int, PMPI_Info_get_valuelen, MPI_Info, char *, int *, int *)
AMPI_CUSTOM_FUNC(int, PMPI_Info_get_nkeys, MPI_Info, int *)
AMPI_CUSTOM_FUNC(int, PMPI_Info_get_nthkey, MPI_Info, int, char *)
AMPI_CUSTOM_FUNC(int, PMPI_Info_dup, MPI_Info, MPI_Info *)
AMPI_CUSTOM_FUNC(int, PMPI_Info_free, MPI_Info *)

AMPI_CUSTOM_FUNC(MPI_Fint, PMPI_Info_c2f, MPI_Info)
AMPI_CUSTOM_FUNC(MPI_Info, PMPI_Info_f2c, MPI_Fint)
#endif

#include "ampiimpl.h"

/*
This file contains function definitions of all MPI functions that are currently
unsupported in AMPI. Calling these functions aborts the application.
*/

#define AMPI_API_NOIMPL(return_type, function_name, ...) \
    AMPI_API_IMPL(return_type, function_name, __VA_ARGS__) \
    { \
        AMPI_API(function_name); \
        CkAbort(STRINGIFY(function_name) " is not implemented in AMPI."); \
    }



/* A.2.2 Datatypes C Bindings */

AMPI_API_NOIMPL(int, MPI_Pack_external, const char datarep[], const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf, MPI_Aint outsize, MPI_Aint *position);
AMPI_API_NOIMPL(int, MPI_Pack_external_size, const char datarep[], int incount, MPI_Datatype datatype, MPI_Aint *size);
// AMPI_API_NOIMPL(MPI_Type_create_darray, int size, int rank, int ndims, const int array_of_gsizes[], const int array_of_distribs[], const int array_of_dargs[], const int array_of_psizes[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype); //provided by ROMIO
// AMPI_API_NOIMPL(MPI_Type_create_subarray, int ndims, const int array_of_sizes[], const int array_of_subsizes[], const int array_of_starts[], int order, MPI_Datatype oldtype, MPI_Datatype *newtype); //provided by ROMIO
AMPI_API_NOIMPL(int, MPI_Unpack_external, const char datarep[], const void *inbuf, MPI_Aint insize, MPI_Aint *position, void *outbuf, int outcount, MPI_Datatype datatype);


/* A.2.6 MPI Environmental Management C Bindings */

AMPI_API_NOIMPL(int, MPI_File_call_errhandler, MPI_File fh, int errorcode);
AMPI_API_NOIMPL(int, MPI_File_create_errhandler, MPI_File_errhandler_function *file_errhandler_fn, MPI_Errhandler *errhandler);
AMPI_API_NOIMPL(int, MPI_File_get_errhandler, MPI_File file, MPI_Errhandler *errhandler);
AMPI_API_NOIMPL(int, MPI_File_set_errhandler, MPI_File file, MPI_Errhandler errhandler);


/* A.2.8 Process Creation and Management C Bindings */

AMPI_API_NOIMPL(int, MPI_Close_port, const char *port_name);
AMPI_API_NOIMPL(int, MPI_Comm_accept, const char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm);
AMPI_API_NOIMPL(int, MPI_Comm_connect, const char *port_name, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *newcomm);
AMPI_API_NOIMPL(int, MPI_Comm_disconnect, MPI_Comm *comm);
AMPI_API_NOIMPL(int, MPI_Comm_get_parent, MPI_Comm *parent);
AMPI_API_NOIMPL(int, MPI_Comm_join, int fd, MPI_Comm *intercomm);
AMPI_API_NOIMPL(int, MPI_Comm_spawn, const char *command, char *argv[], int maxprocs, MPI_Info info, int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]);
AMPI_API_NOIMPL(int, MPI_Comm_spawn_multiple, int count, char *array_of_commands[], char **array_of_argv[], const int array_of_maxprocs[], const MPI_Info array_of_info[], int root, MPI_Comm comm, MPI_Comm *intercomm, int array_of_errcodes[]);
AMPI_API_NOIMPL(int, MPI_Lookup_name, const char *service_name, MPI_Info info, char *port_name);
AMPI_API_NOIMPL(int, MPI_Open_port, MPI_Info info, char *port_name);
AMPI_API_NOIMPL(int, MPI_Publish_name, const char *service_name, MPI_Info info, const char *port_name);
AMPI_API_NOIMPL(int, MPI_Unpublish_name, const char *service_name, MPI_Info info, const char *port_name);


/* A.2.9 One-Sided Communications C Bindings */

AMPI_API_NOIMPL(int, MPI_Win_allocate, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win);
AMPI_API_NOIMPL(int, MPI_Win_allocate_shared, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win);
AMPI_API_NOIMPL(int, MPI_Win_attach, MPI_Win win, void *base, MPI_Aint size);
AMPI_API_NOIMPL(int, MPI_Win_create_dynamic, MPI_Info info, MPI_Comm comm, MPI_Win *win);
AMPI_API_NOIMPL(int, MPI_Win_detach, MPI_Win win, const void *base);
AMPI_API_NOIMPL(int, MPI_Win_flush, int rank, MPI_Win win);
AMPI_API_NOIMPL(int, MPI_Win_flush_all, MPI_Win win);
AMPI_API_NOIMPL(int, MPI_Win_flush_local, int rank, MPI_Win win);
AMPI_API_NOIMPL(int, MPI_Win_flush_local_all, MPI_Win win);
AMPI_API_NOIMPL(int, MPI_Win_lock_all, int assert, MPI_Win win);
AMPI_API_NOIMPL(int, MPI_Win_shared_query, MPI_Win win, int rank, MPI_Aint *size, int *disp_unit, void *baseptr);
AMPI_API_NOIMPL(int, MPI_Win_sync, MPI_Win win);
AMPI_API_NOIMPL(int, MPI_Win_unlock_all, MPI_Win win);


/* A.2.11 I/O C Bindings */

AMPI_API_NOIMPL(int, MPI_CONVERSION_FN_NULL, void *userbuf, MPI_Datatype datatype, int count, void *filebuf, MPI_Offset position, void *extra_state);
AMPI_API_NOIMPL(int, MPI_File_iread_all, MPI_File fh, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
AMPI_API_NOIMPL(int, MPI_File_iread_at_all, MPI_File fh, MPI_Offset offset, void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
AMPI_API_NOIMPL(int, MPI_File_iwrite_all, MPI_File fh, const void *buf, int count, MPI_Datatype datatype, MPI_Request *request);
AMPI_API_NOIMPL(int, MPI_File_iwrite_at_all, MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Request *request);


/* A.2.12 Language Bindings C Bindings */

AMPI_API_NOIMPL(int, MPI_Status_f082f, MPI_F08_status *f08_status, MPI_Fint *f_status);
AMPI_API_NOIMPL(int, MPI_Status_f2f08, MPI_Fint *f_status, MPI_F08_status *f08_status);
AMPI_API_NOIMPL(int, MPI_Type_create_f90_complex, int p, int r, MPI_Datatype *newtype);
AMPI_API_NOIMPL(int, MPI_Type_create_f90_integer, int r, MPI_Datatype *newtype);
AMPI_API_NOIMPL(int, MPI_Type_create_f90_real, int p, int r, MPI_Datatype *newtype);
AMPI_API_NOIMPL(int, MPI_Type_match_size, int typeclass, int size, MPI_Datatype *datatype);
AMPI_API_NOIMPL(MPI_Fint, MPI_Message_c2f, MPI_Message message);
AMPI_API_NOIMPL(MPI_Message, MPI_Message_f2c, MPI_Fint message);
AMPI_API_NOIMPL(int, MPI_Status_c2f, const MPI_Status *c_status, MPI_Fint *f_status);
AMPI_API_NOIMPL(int, MPI_Status_c2f08, const MPI_Status *c_status, MPI_F08_status *f08_status);
AMPI_API_NOIMPL(int, MPI_Status_f082c, const MPI_F08_status *f08_status, MPI_Status *c_status);
AMPI_API_NOIMPL(int, MPI_Status_f2c, const MPI_Fint *f_status, MPI_Status *c_status);


/* A.2.14 Tools / MPI Tool Information Interface C Bindings */

AMPI_API_NOIMPL(int, MPI_T_category_changed, int *stamp);
AMPI_API_NOIMPL(int, MPI_T_category_get_categories, int cat_index, int len, int indices[]);
AMPI_API_NOIMPL(int, MPI_T_category_get_cvars, int cat_index, int len, int indices[]);
AMPI_API_NOIMPL(int, MPI_T_category_get_index, const char *name, int *cat_index);
AMPI_API_NOIMPL(int, MPI_T_category_get_info, int cat_index, char *name, int *name_len, char *desc, int *desc_len, int *num_cvars, int *num_pvars, int *num_categories);
AMPI_API_NOIMPL(int, MPI_T_category_get_num, int *num_cat);
AMPI_API_NOIMPL(int, MPI_T_category_get_pvars, int cat_index, int len, int indices[]);
AMPI_API_NOIMPL(int, MPI_T_cvar_get_index, const char *name, int *cvar_index);
AMPI_API_NOIMPL(int, MPI_T_cvar_get_info, int cvar_index, char *name, int *name_len, int *verbosity, MPI_Datatype *datatype, MPI_T_enum *enumtype, char *desc, int *desc_len, int *bind, int *scope);
AMPI_API_NOIMPL(int, MPI_T_cvar_get_num, int *num_cvar);
AMPI_API_NOIMPL(int, MPI_T_cvar_handle_alloc, int cvar_index, void *obj_handle, MPI_T_cvar_handle *handle, int *count);
AMPI_API_NOIMPL(int, MPI_T_cvar_handle_free, MPI_T_cvar_handle *handle);
AMPI_API_NOIMPL(int, MPI_T_cvar_read, MPI_T_cvar_handle handle, void* buf);
AMPI_API_NOIMPL(int, MPI_T_cvar_write, MPI_T_cvar_handle handle, const void* buf);
AMPI_API_NOIMPL(int, MPI_T_enum_get_info, MPI_T_enum enumtype, int *num, char *name, int *name_len);
AMPI_API_NOIMPL(int, MPI_T_enum_get_item, MPI_T_enum enumtype, int index, int *value, char *name, int *name_len);
AMPI_API_NOIMPL(int, MPI_T_finalize, void);
AMPI_API_NOIMPL(int, MPI_T_init_thread, int required, int *provided);
AMPI_API_NOIMPL(int, MPI_T_pvar_get_index, const char *name, int var_class, int *pvar_index);
AMPI_API_NOIMPL(int, MPI_T_pvar_get_info, int pvar_index, char *name, int *name_len, int *verbosity, int *var_class, MPI_Datatype *datatype, MPI_T_enum *enumtype, char *desc, int *desc_len, int *bind, int *readonly, int *continuous, int *atomic);
AMPI_API_NOIMPL(int, MPI_T_pvar_get_num, int *num_pvar);
AMPI_API_NOIMPL(int, MPI_T_pvar_handle_alloc, MPI_T_pvar_session session, int pvar_index, void *obj_handle, MPI_T_pvar_handle *handle, int *count);
AMPI_API_NOIMPL(int, MPI_T_pvar_handle_free, MPI_T_pvar_session session,MPI_T_pvar_handle *handle);
AMPI_API_NOIMPL(int, MPI_T_pvar_read, MPI_T_pvar_session session, MPI_T_pvar_handle handle,void* buf);
AMPI_API_NOIMPL(int, MPI_T_pvar_readreset, MPI_T_pvar_session session,MPI_T_pvar_handle handle, void* buf);
AMPI_API_NOIMPL(int, MPI_T_pvar_reset, MPI_T_pvar_session session, MPI_T_pvar_handle handle);
AMPI_API_NOIMPL(int, MPI_T_pvar_session_create, MPI_T_pvar_session *session);
AMPI_API_NOIMPL(int, MPI_T_pvar_session_free, MPI_T_pvar_session *session);
AMPI_API_NOIMPL(int, MPI_T_pvar_start, MPI_T_pvar_session session, MPI_T_pvar_handle handle);
AMPI_API_NOIMPL(int, MPI_T_pvar_stop, MPI_T_pvar_session session, MPI_T_pvar_handle handle);
AMPI_API_NOIMPL(int, MPI_T_pvar_write, MPI_T_pvar_session session, MPI_T_pvar_handle handle, const void* buf);

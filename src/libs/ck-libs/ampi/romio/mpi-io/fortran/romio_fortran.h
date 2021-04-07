#undef ADIOI_Malloc
#define ADIOI_Malloc malloc
#undef ADIOI_Free
#define ADIOI_Free free
#undef ADIOI_Strncpy
#define ADIOI_Strncpy strncpy

/* avoids need to compile callers as PIC due to use of stderr */
#ifdef __cplusplus
extern "C"
#endif
void romio_fortran_error_print(const char *);

/*
 * This file is separate from pup_c.h because PUP is included as part
 * of AMPI's extensions to the MPI standard, and certain global variable
 * privatization methods require the AMPI API to be exposed as function pointers
 * through a shim and loader mechanism that needs to list the entire set of
 * provided functions at multiple points in its implementation.
 *
 * See src/libs/ck-libs/ampi/ampi_functions.h for mandatory procedures.
 *
 * For ease of reading: AMPI_CUSTOM_FUNC(ReturnType, FunctionName, Parameters...)
 */

/*Allocate PUP::er of different kind */
AMPI_CUSTOM_FUNC(pup_er, pup_new_sizer, void)
AMPI_CUSTOM_FUNC(pup_er, pup_new_toMem, void *Nbuf)
AMPI_CUSTOM_FUNC(pup_er, pup_new_fromMem, const void *Nbuf)
AMPI_CUSTOM_FUNC(pup_er, pup_new_network_sizer, void)
AMPI_CUSTOM_FUNC(pup_er, pup_new_network_pack, void *Nbuf)
AMPI_CUSTOM_FUNC(pup_er, pup_new_network_unpack, const void *Nbuf)
#if CMK_CCS_AVAILABLE
AMPI_CUSTOM_FUNC(pup_er, pup_new_fmt, pup_er p)
AMPI_CUSTOM_FUNC(void, pup_fmt_sync_begin_object, pup_er p)
AMPI_CUSTOM_FUNC(void, pup_fmt_sync_end_object, pup_er p)
AMPI_CUSTOM_FUNC(void, pup_fmt_sync_begin_array, pup_er p)
AMPI_CUSTOM_FUNC(void, pup_fmt_sync_end_array, pup_er p)
AMPI_CUSTOM_FUNC(void, pup_fmt_sync_item, pup_er p)
#endif
AMPI_CUSTOM_FUNC(void, pup_destroy, pup_er p)

/*Determine what kind of pup_er we have--
return 1 for true, 0 for false.*/
AMPI_CUSTOM_FUNC(int, pup_isPacking, const pup_er p)
AMPI_CUSTOM_FUNC(int, pup_isUnpacking, const pup_er p)
AMPI_CUSTOM_FUNC(int, pup_isSizing, const pup_er p)
AMPI_CUSTOM_FUNC(int, pup_isDeleting, const pup_er p)
AMPI_CUSTOM_FUNC(int, pup_isUserlevel, const pup_er p)
AMPI_CUSTOM_FUNC(int, pup_isCheckpoint, const pup_er p)
AMPI_CUSTOM_FUNC(int, pup_isMigration, const pup_er p)
AMPI_CUSTOM_FUNC(char *, pup_typeString, const pup_er p)

/*Insert a synchronization into the data stream */
AMPI_CUSTOM_FUNC(void, pup_syncComment, const pup_er p, unsigned int sync, const char *message)
AMPI_CUSTOM_FUNC(void, pup_comment, const pup_er p, const char *message)

/*Read the size of a pupper */
AMPI_CUSTOM_FUNC(size_t, pup_size, const pup_er p)

/* Utilities to approximately encode large sizes, within 0.5% */
AMPI_CUSTOM_FUNC(CMK_TYPEDEF_UINT2, pup_encodeSize, size_t s)
AMPI_CUSTOM_FUNC(size_t, pup_decodeSize, CMK_TYPEDEF_UINT2 a)

/*Pack/unpack data items, declared with macros for brevity.
The macros expand like:
AMPI_CUSTOM_FUNC(void, pup_int, pup_er p,int *i) // <- single integer pack/unpack
AMPI_CUSTOM_FUNC(void, pup_ints, pup_er p,int *iarr,int nItems) // <- array pack/unpack
*/
#define PUP_BASIC_DATATYPE(typeName,type) \
  AMPI_CUSTOM_FUNC(void, pup_##typeName, pup_er p,type *v) \
  AMPI_CUSTOM_FUNC(void, pup_##typeName##s, pup_er p,type *arr,size_t nItems)

PUP_BASIC_DATATYPE(char,char)
PUP_BASIC_DATATYPE(short,short)
PUP_BASIC_DATATYPE(int,int)
PUP_BASIC_DATATYPE(long,long)
PUP_BASIC_DATATYPE(uchar,unsigned char)
PUP_BASIC_DATATYPE(ushort,unsigned short)
PUP_BASIC_DATATYPE(uint,unsigned int)
PUP_BASIC_DATATYPE(ulong,unsigned long)
PUP_BASIC_DATATYPE(float,float)
PUP_BASIC_DATATYPE(double,double)
PUP_BASIC_DATATYPE(pointer,void*)
PUP_BASIC_DATATYPE(int8, CMK_TYPEDEF_INT8)
PUP_BASIC_DATATYPE(size_t, size_t)

#undef PUP_BASIC_DATATYPE

/*Pack/unpack untyped byte array:*/
AMPI_CUSTOM_FUNC(void, pup_bytes, pup_er p,void *ptr,size_t nBytes)
FILE *CmiFopen(const char *path, const char *mode);
int CmiFclose(FILE *fp);
void CmiMkdir(const char *dirName);
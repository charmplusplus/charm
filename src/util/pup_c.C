/*
Pack/UnPack Library for UIUC Parallel Programming Lab
Orion Sky Lawlor, olawlor@uiuc.edu, 9/11/2000

This file maps the C bindings for the PUP library onto
the C++ implementation.  This file is C++, but all its 
routines are C-callable.

I realize it looks horrible, but it's just *interface* code--
there's nothing actually happening here.
*/
#include "pup.h"
#include "pup_c.h"
#include "pup_toNetwork.h"
#include "charm-api.h"


/*This maps the opaque C "pup_er p" type to 
a C++ "PUP::er &" type.  We actually want a 
 "*reinterpret_cast<PUP::er *>(p)"
*/
#define mp (*(PUP::er *)p)

/*Allocate PUP::er of different kind */
CDECL pup_er pup_new_sizer()
  { return new PUP::sizer; }
CDECL pup_er pup_new_toMem(void *Nbuf)
  { return new PUP::toMem(Nbuf); }
CDECL pup_er pup_new_fromMem(const void *Nbuf)
  { return new PUP::fromMem(Nbuf); }
CDECL pup_er pup_new_network_sizer()
  { return new PUP_toNetwork_sizer; }
CDECL pup_er pup_new_network_pack(void *Nbuf)
  { return new PUP_toNetwork_pack(Nbuf); }
CDECL pup_er pup_new_network_unpack(const void *Nbuf)
  { return new PUP_toNetwork_unpack(Nbuf); }
#if CMK_CCS_AVAILABLE
#include "ccs-builtins.h"
CDECL pup_er pup_new_fmt(pup_er p)
  { return new PUP_fmt(mp); }
CDECL void pup_fmt_sync_begin_object(pup_er p)
  { mp.synchronize(PUP::sync_begin_object); }
CDECL void pup_fmt_sync_end_object(pup_er p)
  { mp.synchronize(PUP::sync_end_object); }
CDECL void pup_fmt_sync_begin_array(pup_er p)
  { mp.synchronize(PUP::sync_begin_array); }
CDECL void pup_fmt_sync_end_array(pup_er p)
  { mp.synchronize(PUP::sync_end_array); }
CDECL void pup_fmt_sync_item(pup_er p)
  { mp.syncComment(PUP::sync_item); }
#endif
CDECL void pup_destroy(pup_er p)
  { delete ((PUP::er *)p); }

/*Determine what kind of pup_er we have--
return 1 for true, 0 for false.*/
CDECL int pup_isPacking(const pup_er p)
  { return (mp.isPacking())?1:0;}
CDECL int pup_isUnpacking(const pup_er p)
  { return (mp.isUnpacking())?1:0;}
CDECL int pup_isSizing(const pup_er p)
  { return (mp.isSizing())?1:0;}
CDECL int pup_isDeleting(const pup_er p)
  { return (mp.isDeleting())?1:0;}
CDECL int pup_isUserlevel(const pup_er p)
  { return (mp.isUserlevel())?1:0;}
CDECL int pup_isRestarting(const pup_er p)
  { return (mp.isRestarting())?1:0;}
CDECL char* pup_typeString(const pup_er p)
  { return (char *)mp.typeString(); }

FDECL int FTN_NAME(FPUP_ISPACKING,fpup_ispacking)(const pup_er p)
  { return (mp.isPacking())?1:0;}
FDECL int FTN_NAME(FPUP_ISUNPACKING,fpup_isunpacking)(const pup_er p)
  { return (mp.isUnpacking())?1:0;}
FDECL int FTN_NAME(FPUP_ISSIZING,fpup_issizing)(const pup_er p)
  { return (mp.isSizing())?1:0;}
FDECL int FTN_NAME(FPUP_ISDELETING,fpup_isdeleting)(const pup_er p)
  { return (mp.isDeleting())?1:0;}
FDECL int FTN_NAME(FPUP_ISUSERLEVEL,fpup_isuserlevel)(const pup_er p)
  { return (mp.isUserlevel())?1:0;}

/*Read the size of the pupper */
CDECL size_t pup_size(const pup_er p)
  { return mp.size(); }

/*Insert a synchronization into the data stream */
CDECL void pup_syncComment(const pup_er p, unsigned int sync, char *message)
  { mp.syncComment(sync, message); }
/*FDECL void FNT_NAME(FPUP_SYNCCOMMENT,fpup_syncComment)(const pup_er p, unsigned int sync, char *message)
  { mp.syncComment(sync, message); }*/
CDECL void pup_comment(const pup_er p, char *message)
  { mp.comment(message); }

#undef PUP_BASIC_DATATYPE /*from pup_c.h*/
#undef PUP_BASIC_DATATYPEF /*from pup_c.h*/

/*Pack/unpack data items, declared with macros for brevity.
The macros expand like:
void pup_int(pup_er p,int *i) <- single integer pack/unpack
  {(PUP::er & cast p)(*i);}
void pup_ints(pup_er p,int *iarr,size_t nItems) <- array pack/unpack
  {(PUP::er * cast p)(iarr,nItems);}
*/
#define PUP_BASIC_DATATYPE(typeName,type) \
 CDECL void pup_##typeName(pup_er p,type *v) \
   {mp(*v);} \
 CDECL void pup_##typeName##s(pup_er p,type *arr,size_t nItems) \
   {mp(arr,nItems);}

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
PUP_BASIC_DATATYPE(int8,CMK_TYPEDEF_INT8)
PUP_BASIC_DATATYPE(size_t,size_t)

// Pointers have a different signature, so they need special treatment
CDECL void pup_pointer(pup_er p,void **v) {mp(*v,(void*)NULL);}
CDECL void pup_pointers(pup_er p,void **arr,size_t nItems) {mp(arr,nItems,(void*)NULL);}

#define PUP_BASIC_DATATYPEF(typeUP,typelo,type) \
 FDECL void FTN_NAME(FPUP_##typeUP,fpup_##typelo)(pup_er p,type *v) \
   {mp(*v);} \
 FDECL void FTN_NAME(FPUP_##typeUP##SG,fpup_##typelo##sg)(pup_er p,type *arr,size_t *nItems) \
   {mp(arr,*nItems);} \
 FDECL void FTN_NAME(FPUP_##typeUP##S,fpup_##typelo##s)(pup_er p,type *arr,size_t *nItems) \
   {mp(arr,*nItems);}

PUP_BASIC_DATATYPEF(CHAR,char,char)
PUP_BASIC_DATATYPEF(SHORT,short,short)
PUP_BASIC_DATATYPEF(INT,int,int)
PUP_BASIC_DATATYPEF(LONG,long,long)
PUP_BASIC_DATATYPEF(REAL,real,float)
PUP_BASIC_DATATYPEF(DOUBLE,double,double)
PUP_BASIC_DATATYPEF(LOGICAL,logical,int)

/*Pack/unpack untyped byte array:*/
CDECL void pup_bytes(pup_er p,void *ptr,size_t nBytes)
{
  mp((char *)ptr,nBytes);
}

FDECL void FTN_NAME(FPUP_BYTES,fpup_bytes)(pup_er p,void *ptr,size_t *nBytes)
{
  mp((char *)ptr,*nBytes);
}

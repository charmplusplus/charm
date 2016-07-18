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
CDECL pup_er pup_new_sizer(void)
  { return new PUP::sizer; }
CDECL pup_er pup_new_toMem(void *Nbuf)
  { return new PUP::toMem(Nbuf); }
CDECL pup_er pup_new_fromMem(const void *Nbuf)
  { return new PUP::fromMem(Nbuf); }
CDECL pup_er pup_new_network_sizer(void)
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
CDECL int pup_size(const pup_er p)
  { return mp.size(); }

#define SIZE_APPROX_BITS 13

/* Utilities to approximately encode large sizes, within 0.5% */
CDECL CMK_TYPEDEF_UINT2 pup_encodeSize(size_t s)
{
  // Use the top two bits to indicate a scaling factor as a power of 256. At
  // each step up in size, we'll thus lose the bottom 8 bits out of 14:
  // 256/32k < 1%
  CmiUInt2 power = 0;

  while (s > (1UL << SIZE_APPROX_BITS) - 1) {
    power++;
    if (s & (1UL << 6)) // Round up as needed to cut relative error in half
      s += (1UL << 7);
    s >>= 8;
  }

  return (power << SIZE_APPROX_BITS) | s;
}

CDECL size_t pup_decodeSize(CMK_TYPEDEF_UINT2 a)
{
  CmiUInt2 power = a >> SIZE_APPROX_BITS;
  size_t factor = 1UL << (8 * power);

  size_t base = a & ((1UL << SIZE_APPROX_BITS) - 1);

  return base * factor;
}

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
void pup_ints(pup_er p,int *iarr,int nItems) <- array pack/unpack
  {(PUP::er * cast p)(iarr,nItems);}
*/
#define PUP_BASIC_DATATYPE(typeName,type) \
 CDECL void pup_##typeName(pup_er p,type *v) \
   {mp(*v);} \
 CDECL void pup_##typeName##s(pup_er p,type *arr,int nItems) \
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

// Pointers have a different signature, so they need special treatment
CDECL void pup_pointer(pup_er p,void **v) {mp(*v,(void*)NULL);}
CDECL void pup_pointers(pup_er p,void **arr,int nItems) {mp(arr,nItems,(void*)NULL);}

#define PUP_BASIC_DATATYPEF(typeUP,typelo,type) \
 FDECL void FTN_NAME(FPUP_##typeUP,fpup_##typelo)(pup_er p,type *v) \
   {mp(*v);} \
 FDECL void FTN_NAME(FPUP_##typeUP##SG,fpup_##typelo##sg)(pup_er p,type *arr,int *nItems) \
   {mp(arr,*nItems);} \
 FDECL void FTN_NAME(FPUP_##typeUP##S,fpup_##typelo##s)(pup_er p,type *arr,int *nItems) \
   {mp(arr,*nItems);}

PUP_BASIC_DATATYPEF(CHAR,char,char)
PUP_BASIC_DATATYPEF(SHORT,short,short)
PUP_BASIC_DATATYPEF(INT,int,int)
PUP_BASIC_DATATYPEF(LONG,long,long)
PUP_BASIC_DATATYPEF(REAL,real,float)
PUP_BASIC_DATATYPEF(DOUBLE,double,double)
PUP_BASIC_DATATYPEF(LOGICAL,logical,int)


FDECL void FTN_NAME(FPUP_COMPLEX,fpup_complex)(pup_er p, float *v)
{mp(v,2);}

FDECL void FTN_NAME(FPUP_COMPLEXESG,fpup_complexesg)(pup_er p, float *arr, int *nItems)
{mp(arr,2*(*nItems));}

FDECL void FTN_NAME(FPUP_COMPLEXES,fpup_complexes)(pup_er p, float *arr, int *nItems)
{mp(arr,2*(*nItems));}

FDECL void FTN_NAME(FPUP_DOUBLECOMPLEX,fpup_doublecomplex)(pup_er p, double *v)
{mp(v,2);}

FDECL void FTN_NAME(FPUP_DOUBLECOMPLEXESG,fpup_doublecomplexesg)(pup_er p, double *arr, int *nItems)
{mp(arr,2*(*nItems));}

FDECL void FTN_NAME(FPUP_DOUBLECOMPLEXES,fpup_doublecomplexes)(pup_er p, double *arr, int *nItems)
{mp(arr,2*(*nItems));}

/*Pack/unpack untyped byte array:*/
CDECL void pup_bytes(pup_er p,void *ptr,int nBytes)
{
  mp((char *)ptr,nBytes);
}

FDECL void FTN_NAME(FPUP_BYTES,fpup_bytes)(pup_er p,void *ptr,int *nBytes)
{
  mp((char *)ptr,*nBytes);
}

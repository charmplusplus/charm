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

#define C_CALLABLE extern "C" 

/*This maps the opaque C "pup_er *p" type to 
a C++ "PUP::er &" type.  We actually want a 
 "*reinterpret_cast<PUP::er *>(p)"
*/
#define mp (*(PUP::er *)p)

/*Determine what kind of pup_er we have--
return 1 for true, 0 for false.*/
C_CALLABLE int pup_isPacking(const pup_er *p)
  { return (mp.isPacking())?1:0;}
C_CALLABLE int pup_isUnpacking(const pup_er *p)
  { return (mp.isUnpacking())?1:0;}
C_CALLABLE int pup_isSizing(const pup_er *p)
  { return (mp.isSizing())?1:0;}


#undef PUP_BASIC_DATATYPE /*from pup_c.h*/

/*Pack/unpack data items, declared with macros for brevity.
The macros expand like:
void pup_int(pup_er *p,int *i) <- single integer pack/unpack
  {(PUP::er & cast p)(*i);}
void pup_ints(pup_er *p,int *iarr,int nItems) <- array pack/unpack
  {(PUP::er * cast p)(iarr,nItems);}
*/
#define PUP_BASIC_DATATYPE(typeName,type) \
 C_CALLABLE void pup_##typeName(pup_er *p,type *v) \
   {mp(*v);} \
 C_CALLABLE void pup_##typeName##s(pup_er *p,type *arr,int nItems) \
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

/*Pack/unpack untyped byte array:*/
void pup_bytes(pup_er *p,void *ptr,int nBytes)
{
  mp(ptr,nBytes);
}


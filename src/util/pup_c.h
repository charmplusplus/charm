#ifndef _PUP_C_H
#define _PUP_C_H

/*
Pack/UnPack Library for UIUC Parallel Programming Lab
C Bindings version
Orion Sky Lawlor, olawlor@uiuc.edu, 9/11/2000

This library allows you to easily pack an array, structure,
or object into a memory buffer or disk file, and then read 
the object back later.  The library will also handle translating
between different machine representations for integers and floats.

This is the C binding-- the main library is in C++.
From C, you can write a pup routine (the client part of the
library), but you can't make pup_er's (the system part).

Typically, the user has to write separate functions for buffer 
sizing, pack to memory, unpack from memory, pack to disk, and 
unpack from disk.  These functions all perform the exact same function--
namely, they list the members of the array, struct, or object.
Further, all the functions must agree, or the unpacked data will 
be garbage.  This library allows the user to write *one* function,
which will perform all needed packing/unpacking.

A simple example is:
typedef struct foo {
  int x;
  char y;
  unsigned long z;
  float q[3];
} foo;

void pup_foo(pup_er p,foo *f)
{
  pup_int(p,&f->x);
  pup_char(p,&f->y);
  pup_ulong(p,&f->z);
  pup_floats(p,f->q,3);
}

A more complex example is:
typedef struct bar {
  foo f;
  int nArr; <- Length of array below
  double *arr; <- Heap-allocated array
} bar;

void pup_bar(pup_er p,bar *b)
{
  pup_foo(p,&b->f);
  pup_int(p,&b->nArr);
  if (pup_isUnpacking(p))
    b->arr=(double *)malloc(b->nArr*sizeof(double));
  pup_doubles(p,b->arr,b->nArr);
}

*/

#ifdef __cplusplus
extern "C" {
#endif

typedef void *pup_er;

/*Determine what kind of pup_er we have--
return 1 for true, 0 for false.*/
int pup_isPacking(const pup_er p);
int pup_isUnpacking(const pup_er p);
int pup_isSizing(const pup_er p);
int pup_isDeleting(const pup_er p);
int pup_isUserlevel(const pup_er p);

#ifdef FNAME
#undef FNAME
#endif

#if CMK_FORTRAN_USES_TWOSCORE
#  define FNAME(x) x##__
#elif CMK_FORTRAN_USES_ONESCORE
#  define FNAME(x) x##_
#else
#  define FNAME(x) x
#endif

#if CMK_FORTRAN_USES_ALLCAPS
#  define fpup_ispacking    FPUP_ISPACKING
#  define fpup_isunpacking  FPUP_ISUNPACKING
#  define fpup_issizing     FPUP_ISSIZING
#  define fpup_isdeleting   FPUP_ISDELETING
#  define fpup_isuserlevel  FPUP_ISUSERLEVEL
#  define fpup_bytes        FPUP_BYTES
#else
#  define fpup_ispacking    FNAME(fpup_ispacking)
#  define fpup_isunpacking  FNAME(fpup_isunpacking)
#  define fpup_issizing     FNAME(fpup_issizing)
#  define fpup_isdeleting   FNAME(fpup_isdeleting)
#  define fpup_isuserlevel  FNAME(fpup_isuserlevel)
#  define fpup_bytes        FNAME(fpup_bytes)
#endif

int fpup_ispacking(const pup_er p);
int fpup_isunpacking(const pup_er p);
int fpup_issizing(const pup_er p);
int fpup_isdeleting(const pup_er p);
int fpup_isuserlevel(const pup_er p);
void fpup_bytes(pup_er p,void *ptr,int *nBytes);

/*Pack/unpack data items, declared with macros for brevity.
The macros expand like:
void pup_int(pup_er p,int *i); <- single integer pack/unpack
void pup_ints(pup_er p,int *iarr,int nItems); <- array pack/unpack
*/
#define PUP_BASIC_DATATYPE(typeName,type) \
  void pup_##typeName(pup_er p,type *v); \
  void pup_##typeName##s(pup_er p,type *arr,int nItems);

#if CMK_FORTRAN_USES_ALLCAPS
#define PUP_BASIC_DATATYPEF(typeName,type) \
  void FPUP_##typeName(pup_er p,type *v); \
  void FPUP_##typeName##S(pup_er p,type *arr,int *nItems);
PUP_BASIC_DATATYPEF(CHAR,char)
PUP_BASIC_DATATYPEF(SHORT,short)
PUP_BASIC_DATATYPEF(INT,int)
PUP_BASIC_DATATYPEF(REAL,float)
PUP_BASIC_DATATYPEF(DOUBLE,double)
#else
#define PUP_BASIC_DATATYPEF(typeName,type) \
  void FNAME(fpup_##typeName)(pup_er p,type *v); \
  void FNAME(fpup_##typeName##s)(pup_er p,type *arr,int *nItems);
PUP_BASIC_DATATYPEF(char,char)
PUP_BASIC_DATATYPEF(short,short)
PUP_BASIC_DATATYPEF(int,int)
PUP_BASIC_DATATYPEF(real,float)
PUP_BASIC_DATATYPEF(double,double)
#endif

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
void pup_bytes(pup_er p,void *ptr,int nBytes);

#ifdef __cplusplus
};
#endif

#endif

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

#include "conv-config.h"
#include "stddef.h"

/*This is actually a PUP::er *, cast to void *.
  From C++, you can pass "&p" as a pup_er.
*/
typedef void *pup_er;

/*Allocate PUP::er of different kind */
pup_er pup_new_sizer(void);
pup_er pup_new_toMem(void *Nbuf);
pup_er pup_new_fromMem(const void *Nbuf);
pup_er pup_new_network_sizer(void);
pup_er pup_new_network_pack(void *Nbuf);
pup_er pup_new_network_unpack(const void *Nbuf);
#if CMK_CCS_AVAILABLE
pup_er pup_new_fmt(pup_er p);
void pup_fmt_sync_begin_object(pup_er p);
void pup_fmt_sync_end_object(pup_er p);
void pup_fmt_sync_begin_array(pup_er p);
void pup_fmt_sync_end_array(pup_er p);
void pup_fmt_sync_item(pup_er p);
#endif
void pup_destroy(pup_er p);

/*Determine what kind of pup_er we have--
return 1 for true, 0 for false.*/
int pup_isPacking(const pup_er p);
int pup_isUnpacking(const pup_er p);
int pup_isSizing(const pup_er p);
int pup_isDeleting(const pup_er p);
int pup_isUserlevel(const pup_er p);
int pup_isRestarting(const pup_er p);
char *pup_typeString(const pup_er p);

/*Insert a synchronization into the data stream */
void pup_syncComment(const pup_er p, unsigned int sync, char *message);
void pup_comment(const pup_er p, char *message);

/*Read the size of a pupper */
int pup_size(const pup_er p);

/* Utilities to approximately encode large sizes, within 0.5% */
CMK_TYPEDEF_UINT2 pup_encodeSize(size_t s);
size_t pup_decodeSize(CMK_TYPEDEF_UINT2 a);

/*Pack/unpack data items, declared with macros for brevity.
The macros expand like:
void pup_int(pup_er p,int *i); <- single integer pack/unpack
void pup_ints(pup_er p,int *iarr,int nItems); <- array pack/unpack
*/
#define PUP_BASIC_DATATYPE(typeName,type) \
  void pup_##typeName(pup_er p,type *v); \
  void pup_##typeName##s(pup_er p,type *arr,int nItems);

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

/*Pack/unpack untyped byte array:*/
void pup_bytes(pup_er p,void *ptr,int nBytes);

/* These MUST match the sync declarations in pup.h */
enum {
  pup_sync_builtin=0x70000000, /* Built-in, standard sync codes begin here */
  pup_sync_begin=pup_sync_builtin+0x01000000, /* Sync code at start of collection */
  pup_sync_end=pup_sync_builtin+0x02000000, /* Sync code at end of collection */
  pup_sync_last_system=pup_sync_builtin+0x09000000, /* Sync code at end of "system" portion of object */
  pup_sync_array_m=0x00100000, /* Linear-indexed (0..n) array-- use item to separate */
  pup_sync_list_m=0x00200000, /* Some other collection-- use index and item */
  pup_sync_object_m=0x00300000, /* Sync mask for general object */
  
  pup_sync_begin_array=pup_sync_begin+pup_sync_array_m,
  pup_sync_begin_list=pup_sync_begin+pup_sync_list_m, 
  pup_sync_begin_object=pup_sync_begin+pup_sync_object_m, 
  
  pup_sync_end_array=pup_sync_end+pup_sync_array_m, 
  pup_sync_end_list=pup_sync_end+pup_sync_list_m, 
  pup_sync_end_object=pup_sync_end+pup_sync_object_m, 
  
  pup_sync_item=pup_sync_builtin+0x00110000, /* Sync code for a list or array item */
  pup_sync_index=pup_sync_builtin+0x00120000, /* Sync code for index of item in a list */
  
  pup_sync_last
};

#ifdef __cplusplus
}
#endif

#endif

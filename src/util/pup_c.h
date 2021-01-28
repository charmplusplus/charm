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

#include "conv-config.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*This is actually a PUP::er *, cast to void *.
  From C++, you can pass "&p" as a pup_er.
*/
typedef void *pup_er;


#ifndef AMPI_INTERNAL_SKIP_FUNCTIONS

#define AMPI_CUSTOM_FUNC(return_type, function_name, ...) \
extern return_type function_name(__VA_ARGS__);

#include "pup_c_functions.h"

#undef AMPI_CUSTOM_FUNC

#endif /* !defined AMPI_INTERNAL_SKIP_FUNCTIONS */


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

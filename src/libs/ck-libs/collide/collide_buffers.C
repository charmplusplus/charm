/*
Ancient, hideous custom memory management classes.
FIXME: replace these with something well-tested,
standard, and modern, like std::vector.

Orion Sky Lawlor, olawlor@acm.org, 2001/2/5
*/
#include <stdio.h>
#include <stdlib.h> /* for malloc, free */
#include <string.h> /* for memcpy */
#include "collide_buffers.h"
#include "charm++.h"

/************** MemoryBuffer ****************
Manages an expandable buffer of bytes.  Like std::vector,
but typeless.
*/

memoryBuffer::memoryBuffer()//Empty initial buffer
{
  data=NULL;len=0;
}

memoryBuffer::memoryBuffer(size_t initLen)//Initial capacity specified
{
  data=NULL;len=0;reallocate(initLen);
  //printf("COLLIDE: Initializing memory buffer to size %d\n", initLen);
}

memoryBuffer::~memoryBuffer()//Deletes array
{
  //printf("COLLIDE: Freeing memory buffer of size %d\n", len);
  free(data);
}

void memoryBuffer::setData(const void *toData,size_t toLen)//Reallocate and copy
{
  //printf("COLLIDE: Setting memory buffer size from %d to %d\n", len, toLen);
  reallocate(toLen);
  memcpy(data,toData,toLen);
}

void memoryBuffer::resize(size_t newlen)//Reallocate, preserving old data
{
  if (len==0) {reallocate(newlen); return;}
  if (len==newlen) return;
  //printf("COLLIDE: Resizing memory buffer from size %d to %d\n", len, newlen);
  void *oldData=data; size_t oldlen=len;
  data=malloc(len=newlen);
  memcpy(data,oldData,oldlen<newlen?oldlen:newlen);
  free(oldData);
}

void memoryBuffer::reallocate(size_t newlen)//Free old data, allocate new
{
  free(data);
  data=malloc(len=newlen);
}


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
}
memoryBuffer::~memoryBuffer()//Deletes array
{
	free(data);
}
void memoryBuffer::setData(const void *toData,size_t toLen)//Reallocate and copy
{
	reallocate(toLen);
	memcpy(data,toData,toLen);
}
void memoryBuffer::resize(size_t newlen)//Reallocate, preserving old data
{
	if (len==0) {reallocate(newlen); return;}
	if (len==newlen) return;
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


/*
Pack/UnPack Library for UIUC Parallel Programming Lab
Orion Sky Lawlor, olawlor@uiuc.edu, 4/5/2000

This library allows you to easily pack an array, structure,
or object into a memory buffer or disk file, and then read 
the object back later.  The library will also handle translating
between different machine representations.

This file is needed because virtual function definitions in
header files cause massive code bloat-- hence the PUP library
virtual functions are defined here.

*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pup.h"

PUP::er::~er() {} //<- might be needed by some child
bool PUP::er::isSizing(void) const {return false;}
bool PUP::er::isPacking(void) const {return false;}
bool PUP::er::isUnpacking(void) const {return false;}

bool PUP::packer::isPacking(void) const {return true;}
bool PUP::unpacker::isUnpacking(void) const {return true;}

void PUP::sizer::bytes(void * /*p*/,int n,size_t itemSize,dataType /*t*/,const char *desc)
	{nBytes+=n*itemSize;}
bool PUP::sizer::isSizing(void) const {return true;}

void PUP::toMem::bytes(void *p,int n,size_t itemSize,dataType /*t*/,const char *desc)
	{n*=itemSize; memcpy((void *)buf,p,n); buf+=n;}
void PUP::fromMem::bytes(void *p,int n,size_t itemSize,dataType /*t*/,const char *desc)
	{n*=itemSize; memcpy(p,(const void *)buf,n); buf+=n;}

void PUP::toDisk::bytes(void *p,int n,size_t itemSize,dataType /*t*/,const char *desc)
	{fwrite(p,itemSize,n,outF);}
void PUP::fromDisk::bytes(void *p,int n,size_t itemSize,dataType /*t*/,const char *desc)
	{fread(p,itemSize,n,inF);}

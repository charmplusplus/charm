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
#include "converse.h"
#include "pup.h"

PUP::er::~er() {}

void PUP::sizer::bytes(void * /*p*/,int n,size_t itemSize,dataType /*t*/,const char *desc)
	{nBytes+=n*itemSize;}

/*Memory PUP::er's*/
void PUP::toMem::bytes(void *p,int n,size_t itemSize,dataType /*t*/,const char *desc)
  {n*=itemSize; memcpy((void *)buf,p,n); buf+=n;}
void PUP::fromMem::bytes(void *p,int n,size_t itemSize,dataType /*t*/,const char *desc)
  {n*=itemSize; memcpy(p,(const void *)buf,n); buf+=n;}

/*Disk PUP::er's*/
PUP::disk::~disk() 
	{fclose(F);}
void PUP::toDisk::bytes(void *p,int n,size_t itemSize,dataType /*t*/,const char *desc)
	{fwrite(p,itemSize,n,F);}
void PUP::fromDisk::bytes(void *p,int n,size_t itemSize,dataType /*t*/,const char *desc)
	{fread(p,itemSize,n,F);}

/*************************************
For seeking:
Occasionally, one will need to pack and unpack items in different
orders (e.g., pack the user data, then the runtime support; but
unpack the runtime support first, then the user data).  These routines
support this, via the "PUP::seekBlock" class.

The abstraction is a (nestable) "seek block", which may contain
several "seek sections".  A typical use is:
//Code:
	PUP::seekBlock s(p,2);
	if (p.isUnpacking()) {s.seek(0); rt.pack(p); }
	s.seek(1); ud.pack(p); 
	if (p.isPacking()) {s.seek(0); rt.pack(p); }
	s.endBlock();
*/
PUP::seekBlock::seekBlock(PUP::er &Np,int nSections)
	:nSec(nSections),p(Np) 
{
	if (nSections<0 || nSections>maxSections)
		CmiAbort("Invalid # of sections passed to PUP::seekBlock!");
	p.impl_startSeek(*this);
	if (p.isPacking()) 
	{ //Must fabricate the section table
		secTabOff=p.impl_tell(*this);
		for (int i=0;i<=nSec;i++) secTab[i]=-1;
	}
	p(secTab,nSec+1);
	hasEnded=false;
}
PUP::seekBlock::~seekBlock() 
{
	if (!hasEnded)
		endBlock();
}

void PUP::seekBlock::seek(int toSection) 
{
	if (toSection<0 || toSection>=nSec)
		CmiAbort("Invalid section # passed to PUP::seekBlock::seek!");
	if (p.isPacking()) //Build the section table
		secTab[toSection]=p.impl_tell(*this);
	else if (p.isUnpacking()) //Extract the section table
		p.impl_seek(*this,secTab[toSection]);
	/*else ignore the seeking*/
}

void PUP::seekBlock::endBlock(void) 
{
	if (p.isPacking()) {
		//Finish off and write out the section table
		secTab[nSec]=p.impl_tell(*this);
		p.impl_seek(*this,secTabOff);
		p(secTab,nSec+1); //Write out the section table
	}
	//Seek to the end of the seek block
	p.impl_seek(*this,secTab[nSec]);
	p.impl_endSeek(*this);
	hasEnded=true;
}

/** PUP::er seek implementation routines **/
/*Default seek implementations are empty, which is the 
appropriate behavior for, e.g., sizers.
*/
void PUP::er::impl_startSeek(PUP::seekBlock &s) /*Begin a seeking block*/
{}
int PUP::er::impl_tell(seekBlock &s) /*Give the current offset*/
{return 0;}
void PUP::er::impl_seek(seekBlock &s,int off) /*Seek to the given offset*/
{}
void PUP::er::impl_endSeek(seekBlock &s)/*End a seeking block*/
{}


/*Memory buffer seeking is trivial*/
void PUP::mem::impl_startSeek(seekBlock &s) /*Begin a seeking block*/
  {s.data.ptr=buf;}
int PUP::mem::impl_tell(seekBlock &s) /*Give the current offset*/
  {return buf-s.data.ptr;}
void PUP::mem::impl_seek(seekBlock &s,int off) /*Seek to the given offset*/
  {buf=s.data.ptr+off;}

/*Disk buffer seeking is also simple*/
void PUP::disk::impl_startSeek(seekBlock &s) /*Begin a seeking block*/
  {s.data.loff=ftell(F);}
int PUP::disk::impl_tell(seekBlock &s) /*Give the current offset*/
  {return (int)(ftell(F)-s.data.loff);}
void PUP::disk::impl_seek(seekBlock &s,int off) /*Seek to the given offset*/
  {fseek(F,s.data.loff+off,0);}

/*PUP::xlater just forwards seek calls to its unpacker.*/
void PUP::xlater::impl_startSeek(seekBlock &s) /*Begin a seeking block*/
  {myUnpacker.impl_startSeek(s);}
int PUP::xlater::impl_tell(seekBlock &s) /*Give the current offset*/
  {return myUnpacker.impl_tell(s);}
void PUP::xlater::impl_seek(seekBlock &s,int off) /*Seek to the given offset*/
  {myUnpacker.impl_seek(s,off);}
void PUP::xlater::impl_endSeek(seekBlock &s) /*Finish a seeking block*/
  {myUnpacker.impl_endSeek(s);}

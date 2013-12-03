
#ifndef PUP_CMIALLOC_H__
#define PUP_CMIALLOC_H__

#include "pup.h"
#include "converse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/********  CMIALLOC buffer management functions ******/

/* Given a user chunk m, extract the enclosing chunk header fields: */

//Align data sizes to 8 bytes
#define ALIGN8_LONG(x)       (long)((~7)&((x)+7))

//Assuming Size of CmiChunkHeader is a multiple of 8 bytes!!

/*For CMI alloc'ed memory
  CmiAlloc currently has the following memory footprint

  |int size|int ref|user allocated buffer|
  
  The ref count can be set so that a sub buffers can be individually
  freed.  When all sub buffers have been individually freed the
  super buffer is freed. This is very use ful for message combining.

  For example, the memory foot print of a combined message could
  look like this

    |size_bigbuf|ref_bigbuf|begin_bigbuf
     ..... |size_subbuf1|ref_subbuf1|subbuf1|
     .... |size_subbuf2|ref_subbuf2|subbuf2|
     ..... |size_subbufk|ref_subbufk|subbufk|

  The usr program can then get use each of the sub_bufs
  sub_buf1, ... , sub_bufk

  These sub_bufs could be converse messages or just data pointers

  To create such a combined message, the pup framwork can be used

    PUP_cmiAllocSizer psizer;
    for(count = 0; count < k; count++)
        psizer.pupCmiAllocBuf(&sub_buf[k]);

    void *bigbuf = CmiAlloc(psizer.size());
    PUP_toCmiAllocMem pmem(bigbuf);

    for(count = 0; count < k; count++)
        pmem.pupCmiAllocBuf(&sub_buf[k]);

    //NOW big buf has all the subbuffers
    //If you also add a converse header to it it can become a 
    //converse message

    //To extract sub buffers from bigbuf
    PUP_fromCmiAllocMem pfmem(bigbuf);

    //Notice not memory allocation or copying needed on the
    //destination
    for(count = 0; count < k; count++)
        pfmem.pupCmiAllocBuf(&sub_buf[k]);


  To free these buffers the user program on the destination after
  receiving the messages MUST CALL!!

    CmiFree(sub_buf[1]);
    ....
    CmiFree(sub_buf[k]);
  */

/**
   The current test for this code is in src/ck-com/MsgPacker.C
   I will later port CmiMultipleSend
**/

class PUP_cmiAllocSizer : public PUP::sizer {
 protected:
    //Generic bottleneck: n items of size itemSize
    virtual void bytes(void *p,int n,size_t itemSize,PUP::dataType t);
 public:
    //Write data to the given buffer
    PUP_cmiAllocSizer(void): PUP::sizer() {}
        
    //Must be a CmiAlloced buf while packing
    void pupCmiAllocBuf(void **msg);
        
    //In case source is not CmiAlloced the size can be passed and any
    //user buf can be converted into a cmialloc'ed buf
    void pupCmiAllocBuf(void **msg, int msg_size);
};
 

//For packing into a preallocated, presized memory cmialloc'ed buffer
//Can use reference counting to reduce one level of copying on the
//receiver
class PUP_toCmiAllocMem : public PUP::toMem {
 protected:
    //Generic bottleneck: pack n items of size itemSize from p.
    virtual void bytes(void *p,int n,size_t itemSize, PUP::dataType t);
    
 public:
    //Write data to the given buffer
    PUP_toCmiAllocMem(int size): PUP::toMem(CmiAlloc(size)) {}
    PUP_toCmiAllocMem(void *buf): PUP::toMem(buf) {}
    
    //Copy the size of the buffer and the reference count while packing
    //Get the buffer directly from the message while unpacking
    //Saves on a copy
    void pupCmiAllocBuf(void **msg);
    
    //Non cmialloc'ed buffers can also be passed and pupped as a
    //cmialloc'ed buffers
    void pupCmiAllocBuf(void **msg, int size);
};
 
 
//For unpacking from a memory buffer
class PUP_fromCmiAllocMem : public PUP::fromMem {
 protected:
    //Generic bottleneck: unpack n items of size itemSize from p.
    virtual void bytes(void *p,int n,size_t itemSize, PUP::dataType t);
 public:
    //Read data from the given buffer
    //The buffer SHOULD have been CMIALLOC'ed
    PUP_fromCmiAllocMem(const void *Nbuf): PUP::fromMem(Nbuf) {}
    
    //Copy the size of the buffer and the reference count while packing
    //Get the buffer directly from the message while unpacking
    //Saves on a copy
    void pupCmiAllocBuf(void **msg);
    
    //size is irrelevant and for consistency with toCmiAllocMem
    void pupCmiAllocBuf(void **msg, int size) {
        pupCmiAllocBuf(msg);
    }
};

#endif

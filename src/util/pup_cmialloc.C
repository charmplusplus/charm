#include "pup.h"
#include "pup_cmialloc.h"
#include "converse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/********  CMIALLOC buffer management functions ******/

/* Given a user chunk m, extract the enclosing chunk header fields: */
#define SIZEFIELD(m) (((CmiChunkHeader *)(m))[-1].size)
#define REFFIELD(m) (((CmiChunkHeader *)(m))[-1].ref)
#define BLKSTART(m) (((CmiChunkHeader *)(m))-1)

//Align data sizes to 8 bytes
#define ALIGN8(x)       (int)((~7)&((x)+7))

//Assuming Size of CmiChunkHeader is a multiple of 8 bytes!!

void PUP_cmiAllocSizer::bytes(void *,int n,size_t itemSize,PUP::dataType) {
    nBytes += n * itemSize;
}

void PUP_cmiAllocSizer::pupCmiAllocBuf(void **msg) {
    CmiChunkHeader chnk_hdr = *(BLKSTART(*msg));
    pupCmiAllocBuf(msg, chnk_hdr.size);
}

void PUP_cmiAllocSizer::pupCmiAllocBuf(void **msg, int msg_size) {

    //The cmialloced buf can only start at an aligned memory location
    //So nbytes has to be aligned
    nBytes = ALIGN8(nBytes);

    nBytes += sizeof(CmiChunkHeader);
    //Here the user buffer pointer will start, hence everything has to
    //be aligned till here
    nBytes += msg_size;  //The actual size of the user message
}


void PUP_toCmiAllocMem::bytes(void *p, int n, size_t itemSize, 
                              PUP::dataType t) {

    n *= itemSize;
    memcpy((void *)buf, p, n);    
    buf += n;
}

void PUP_toCmiAllocMem::pupCmiAllocBuf(void **msg) {
    pupCmiAllocBuf(msg, SIZEFIELD(msg));
}

void PUP_toCmiAllocMem::pupCmiAllocBuf(void **msg, int msg_size) {

    CmiChunkHeader chnk_hdr;

    buf = origBuf + ALIGN8(size());

    chnk_hdr.size = msg_size;
    chnk_hdr.ref = origBuf - (buf + sizeof(CmiChunkHeader));
    
    //Copy the Chunk header
    memcpy(buf, &chnk_hdr, sizeof(CmiChunkHeader));
    buf += sizeof(CmiChunkHeader);

    //Now buf is a memory aligned pointer
    //Copy the message
    //While unpacking, this aligned pointer will be returned
    memcpy(buf, *msg, msg_size);
    buf += msg_size;
}


void PUP_fromCmiAllocMem::bytes(void *p, int n, size_t itemSize, 
                                PUP::dataType t)
{
    n*=itemSize;
    memcpy(p,(const void *)buf,n);
    
    buf+= n;
}


void PUP_fromCmiAllocMem::pupCmiAllocBuf(void **msg) {
    //First align buf
    buf = (PUP::myByte *)ALIGN8((long)buf);

    //Now get the chunk header
    CmiChunkHeader chnk_hdr;    
    //Get the Chunk header
    memcpy(&chnk_hdr, buf, sizeof(CmiChunkHeader));
    buf += sizeof(CmiChunkHeader);

    //Now we are at the begining of the user buffer
    *msg = buf;

    //Move the local buf forward by size bytes
    buf += chnk_hdr.size;
    
    //update the reference count of the original buf
    REFFIELD(origBuf) ++;
}

/***** END CmiAlloc'ed buffer management functions ***********/

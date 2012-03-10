/**
   @addtogroup ConvComlib
   @{
   @file 
   @brief Stores lists of messages and sizes from multiple PEs

   This header is meant for usage with ExtractAndVectorize and
   ExtractAndVectorizeAll. It will contain a list of messages with its sizes.
   The parent class will later call a CmiFree to sizes and msgs. This will
   delete this two arrays, and also the containint ptvectorlist struct. This is
   done (in the two functions) by allocating a single message containing both
   the ptvectorlist struct, and the two arrays. Throught CmiReference
   (incremented only once), when both the arrays are deleted, the struct will
   also dirappear.
   
*/

#ifndef PETABLE_H
#define PETABLE_H

#include "router.h"

#ifndef NULL
#define NULL 0
#endif

#define CMK_COMLIB_USE_VECTORIZE 0

#define MSGQLEN 32

typedef struct ptinfo {
  int refCount;
  int magic;
  int offset;
  /*int freelistindex;*/
  int msgsize;
  void *msg;
  struct ptinfo * next;
} PTinfo;

typedef struct ptvectorlist {
  int count;
  int *sizes;
  char **msgs;
}* PTvectorlist;



/* Reduce the no. of mallocs by allocating from
 * a free list. By allocating 21 at a time, it allocates
 * 512 contiguous bytes. */
#define PTALLOC(ktmp) {\
  if (PTFreeList) {\
  	ktmp=PTFreeList;\
	PTFreeList=ktmp->next;\
  }\
  else {\
  	ktmp=(PTinfo *)CmiAlloc(21*sizeof(PTinfo)+sizeof(PTinfo *));\
        for (int ii=1; ii<20; ++ii) {\
          ktmp[ii].next = &(ktmp[ii+1]);\
        }\
        ktmp[20].next = NULL;\
        PTFreeList=&(ktmp[1]);\
        *((PTinfo**)(&ktmp[21]))=PTFreeChunks;\
        PTFreeChunks=ktmp;\
  }\
}

#define PTFREE(ktmp) {\
  ktmp->next=PTFreeList;\
  PTFreeList=ktmp;\
}

#define PTNEXTCHUNK(ktmp)  (*((PTinfo**)(&ktmp[21])));

#define REALLOC(ktmp, ksize) {\
   PTinfo **junkptr=(PTinfo **)CmiAlloc(2*ksize*sizeof(void *));\
   for (int ki=0; ki<ksize;ki++) junkptr[ki]=ktmp[ki];\
   CmiFree(ktmp);\
   ktmp=junkptr;\
}

class PeTable {
  private:
    PTinfo ***PeList;
    CkVec<PTinfo *> ptrvec;

    PTinfo *PTFreeList;
    PTinfo *PTFreeChunks;
    //	char * CombBuffer;
    int *msgnum, *MaxSize;
    int NumPes;
    int magic;
    //GList *FreeList;

    inline int TotalMsgSize(int npe, int *pelist, int *nm, int *nd) {
        register int totsize=0;
        magic++;
        *nm=0;
        *nd=0;        
        
        for (int i=0;i<npe;i++) {            
            int index = pelist[i];            
            *nm += msgnum[index];
            
            ComlibPrintf("%d: NUM MSGS %d, %d\n", CkMyPe(), index, 
                         msgnum[index]);
            
            for (int j=0;j<msgnum[index];j++) {
                if (PeList[index][j]->magic != magic) {                    
                    int tmp_size = PeList[index][j]->msgsize;
                    tmp_size = ALIGN8(tmp_size);                
                    totsize += tmp_size;                
                    totsize += sizeof(int)+sizeof(int);                    
                    PeList[index][j]->magic=magic;
                    (*nd)++;
                }
            }
        }
        return(totsize);
    }

 public:
    
    PeTable(int n);
    ~PeTable();
    
    inline void InsertMsgs(int npe, int *pelist, int size, void *msg) {
        PTinfo *tmp;
        PTALLOC(tmp);
        tmp->refCount=0;
        tmp->magic=0;
        tmp->offset=0;
        /*tmp->freelistindex=-1;*/
        tmp->msgsize=size;
        tmp->msg=msg;
        
        for (int j=0;j<npe;j++) {
            tmp->refCount++;
            int index=pelist[j];
            
            ComlibPrintf("[%d] Inserting %d %d %d\n", CkMyPe(), 
                         msgnum[index], index, size);
            
            if (msgnum[index] >= MaxSize[index]) {
                REALLOC(PeList[index], MaxSize[index]);
                MaxSize[index] *= 2;
            }
            PeList[index][msgnum[index]]=tmp;
            msgnum[index]++;
        }
    }

    inline void InsertMsgs(int npe, int *pelist, int nmsgs, void **msglist){
        msgstruct **m=(msgstruct **)msglist;
        for (int i=0;i<nmsgs;i++)
            InsertMsgs(npe, pelist, m[i]->msgsize, m[i]->msg);
    }
        
    void ExtractAndDeliverLocalMsgs(int pe, Strategy *myStrat);
    
    int UnpackAndInsert(void *in);
    int UnpackAndInsertAll(void *in, int npes, int *pelist);
    
    char * ExtractAndPack(comID, int, int, int *pelist, int *length); 
    char * ExtractAndPackAll(comID id, int ufield, int *length);
    
    struct ptvectorlist * ExtractAndVectorize(comID, int, int, int *pelist); 
    struct ptvectorlist * ExtractAndVectorizeAll(comID id, int ufield);
    
    void GarbageCollect();
    void Purge();
};

#endif

/*@}*/

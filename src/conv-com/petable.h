/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef PETABLE_H
#define PETABLE_H

#ifndef NULL
#define NULL 0
#endif

#define MSGQLEN 32

typedef struct ptinfo {
  int refCount;
  int magic;
  int offset;
  int freelistindex;
  int msgsize;
  void *msg;
  struct ptinfo * next;
} PTinfo;

typedef struct {
  int refCount;
  int flag;
  void * ptr;
} InNode;

class GList {
 private:
	InNode *InList;
	int InListIndex;
 public:
	GList();
	~GList();
	int AddWholeMsg(void *);
	void setRefcount(int, int);
	void DeleteWholeMsg(int);
	void DeleteWholeMsg(int, int);
	void GarbageCollect();
	void Add(void *);
	void Delete();
};

#define ALIGN8(x)       (int)((~7)&((x)+7))

/* Reduce the no. of mallocs by allocating from
 * a free list */
#define PTALLOC(ktmp) {\
  if (PTFreeList) {\
  	ktmp=PTFreeList;\
	PTFreeList=ktmp->next;\
  }\
  else {\
  	ktmp=(PTinfo *)CmiAlloc(sizeof(PTinfo));\
	}\
}

#define PTFREE(ktmp) {\
  ktmp->next=PTFreeList;\
  PTFreeList=ktmp;\
}

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
    //	char * CombBuffer;
    int *msgnum, *MaxSize;
    int NumPes;
    int magic;
    GList *FreeList;

    inline int TotalMsgSize(int npe, int *pelist, int *nm, int *nd) {
        register int totsize=0;
        magic++;
        *nm=0;
        *nd=0;        
        
        for (int i=0;i<npe;i++) {            
            int index = pelist[i];            
            *nm += msgnum[index];
            
            ComlibPrintf("%d: NUM MSGS %d, %d\n", CmiMyPe(), index, 
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
        tmp->freelistindex=-1;
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
        
    void ExtractAndDeliverLocalMsgs(int pe);
    
    int UnpackAndInsert(void *in);
    int UnpackAndInsertAll(void *in, int npes, int *pelist);
    
    char * ExtractAndPack(comID, int, int, int *pelist, int *length); 
    char * ExtractAndPackAll(comID id, int ufield, int *length);
    
    void GarbageCollect();
    void Purge();
};

#endif

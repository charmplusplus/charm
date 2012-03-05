/**
   @addtogroup ConvComlib
   @{
   @file 
   @brief Tables of messages for PEs 
*/

#include <string.h>
#include <stdlib.h>
//#include <converse.h>
//#include "convcomlib.h"
#include "petable.h"
//#include "converse.h"

#define BIGBUFFERSIZE 65536
#define PTPREALLOC    100

struct AllToAllHdr {
  char dummy[CmiReservedHeaderSize];
  int refno;
  comID id;
  int size;
  int ufield;
  int nmsgs;
};


/**************************************************************
 * Preallocated memory=P*MSGQLEN ptr + 2P ints + 1000 ptrs
 **************************************************************/
PeTable :: PeTable(int n) {
  NumPes=n;
  magic=0;
  PeList = (PTinfo ***)CmiAlloc(sizeof(PTinfo *)*NumPes);
  //  ComlibPrintf("Pelist[%d][%d]\n", NumPes, MSGQLEN);
  msgnum=new int[NumPes];
  MaxSize=new int[NumPes];
  for (int i=0;i<NumPes;i++) {
    msgnum[i]=0;
    MaxSize[i]=MSGQLEN;
    PeList[i]=(PTinfo **)CmiAlloc(sizeof(PTinfo *)*MSGQLEN);
    for (int j=0;j<MSGQLEN;j++) PeList[i][j]=0;
  }

  //ptrlist=(PTinfo **)CmiAlloc(1000*sizeof(PTinfo *));
  //  FreeList= new GList;
  //CombBuffer=(char *)CmiAlloc(BIGBUFFERSIZE);

  PTFreeList=NULL;
  PTFreeChunks=NULL;
}

PeTable :: ~PeTable() {
  int i;
  for (i=0;i<NumPes;i++) CmiFree(PeList[i]);
  CmiFree(PeList);
  delete msgnum;
  delete MaxSize;
  GarbageCollect();
  //CmiFree(ptrlist);
  PTinfo *tmp;
  while (PTFreeChunks) {
    tmp=PTFreeChunks;
    PTFreeChunks=PTNEXTCHUNK(tmp);
    CmiFree(tmp);
  }
  // delete FreeList;

}

void PeTable:: Purge() {
  for (int i=0; i<NumPes;i++) {
    if (msgnum[i]) {
      // ComlibPrintf("%d Warning: %d Undelivered Messages for %d\n", CkMyPe(), msgnum[i], i);
      //msgnum[i]=0;
    }
  }
  GarbageCollect();
  //  ComlibPrintf("combcount = %d\n", combcount);
  //combcount = 0;
}

void PeTable :: ExtractAndDeliverLocalMsgs(int index, Strategy *myStrat) {
  int j;
  msgstruct m;

  ComlibPrintf("%d:Delivering %d local messages\n", CkMyPe(), msgnum[index]);
  for (j=msgnum[index]-1;(j>=0);j--) {

    m.msgsize=PeList[index][j]->msgsize;
    m.msg=PeList[index][j]->msg;

    if (--(PeList[index][j]->refCount) <=0) {
      //CmiSyncSendAndFree(CkMyPe()/*index*/, m.msgsize, (char*)m.msg);
      myStrat->deliver((char*)m.msg, m.msgsize);
      PTFREE(PeList[index][j]);
    }
    else {
      char *dupmsg = (char*)CmiAlloc(m.msgsize);
      memcpy(dupmsg, m.msg, m.msgsize);
      myStrat->deliver(dupmsg, m.msgsize);
      //CmiSyncSend(CkMyPe()/*index*/, m.msgsize, (char*)m.msg);
    }
    PeList[index][j]=NULL;
  }
  msgnum[index]=j+1;

  return;
}


#undef PACK
#undef PACKMSG
//#define PACKINT(data) {((int*)t)[0] = data; t+=sizeof(int);}
#define PACK(type,data) {junk=(char *)&(data); memcpy(t, junk, sizeof(type)); t+=sizeof(type);}
#define PACKMSG(data, size) { memcpy(p+msg_offset, (data), size); msg_offset += size; }

/*
  Protocol:
  |     AllToAllHdr      |npe|ChunkHdr1|msg1|ChunkHdr2|msg2|...
  |ref|comid|ufield|nmsgs|
*/
char * PeTable ::ExtractAndPack(comID id, int ufield, int npe, 
                                int *pelist, int *length) {
  char *junk;
  int nummsgs, offset, num_distinctmsgs;
        
  int tot_msgsize=TotalMsgSize(npe, pelist, &nummsgs, &num_distinctmsgs);

  ComlibPrintf("%d In ExtractAndPack %d, %d\n", CmiMyPe(), npe, nummsgs); 

  if (tot_msgsize ==0) {
    *length=0;
        
    ComlibPrintf("Returning NULL\n");
    return(NULL);
  }
    
  int msg_offset = sizeof(struct AllToAllHdr) + (npe + nummsgs + 1) * sizeof(int);
  //int msg_offset = CmiReservedHeaderSize + sizeof(comID) 
  //    + (npe + 4 + nummsgs) * sizeof(int);  

  msg_offset = ALIGN8(msg_offset);
    
  *length = tot_msgsize;
  *length += msg_offset;
  char *p;
  p=(char *)CmiAlloc(*length);

  char *t = p + CmiReservedHeaderSize;
  int i, j;
  if (!p) CmiAbort("Big time problem\n");
  magic++;

  int refno = id.refno;    

  PACK(int, refno);
  PACK(comID, id);
  PACK(int, *length);
  PACK(int, ufield);
  PACK(int, nummsgs);
  PACK(int, npe);
    
  int lesspe=0;
  int npacked = 0;
  for (i=0;i<npe;i++) {
    int index=pelist[i];

    if (msgnum[index]<=0) {
      lesspe++;
            
      ComlibPrintf("[%d] msgnum[index]<=0 !!!!!\n", CkMyPe());
      continue;
    }
        
    ComlibPrintf("%d Packing pelist[%d]\n", CkMyPe(), index);
    register int newval=-1*pelist[i];
    PACK(int, newval); 

    for (j=0;j<msgnum[index];j++) {
      if (PeList[index][j]->magic == magic) {
	offset=(PeList[index][j]->offset);
      }
      else {
	npacked ++;
                
	//offset=msg_offset;
	offset=npacked;
	PeList[index][j]->magic=magic;
	PeList[index][j]->offset=offset;
	PTinfo *tempmsg=PeList[index][j];
 		
	CmiChunkHeader hdr;
	hdr.size = tempmsg->msgsize;
	hdr.ref = -1;
	PACKMSG(&hdr, sizeof(CmiChunkHeader));
	PACKMSG(tempmsg->msg, tempmsg->msgsize);

	msg_offset = ALIGN8(msg_offset);
      }
            
      //ComlibPrintf("%d Packing msg_offset=%d\n", CkMyPe(), offset);
      PACK(int, offset); 

      if (--(PeList[index][j]->refCount) <=0) {
	CmiFree(PeList[index][j]->msg);
                
	PTFREE(PeList[index][j]);
      }
      PeList[index][j]=NULL;
    }
    msgnum[index]=0;
  }
  //offset=-1;
  //PACK(int, offset);

  /*    
	if (lesspe) {
        t=p+CmiReservedHeaderSize+2*sizeof(int) + sizeof(comID);
	npe=npe-lesspe;
	PACK(int, npe);
	}
  */

  return(p);
} 

/*Used for all to all multicast operations.  Assumes that each message
  is destined to all the processors, to speed up all to all
  substantially --Sameer 09/03/03 
  
  Protocol:
  |     AllToAllHdr      |ChunkHdr1|msg1|ChunkHdr2|msg2|...
  |ref|comid|ufield|nmsgs|
*/

char * PeTable ::ExtractAndPackAll(comID id, int ufield, int *length) {
  int nmsgs = 0;
  int i, j;
  int index = 0;

  ComlibPrintf("[%d] In Extract And Pack All\n", CkMyPe());

  //Increment magic to detect duplicate messages
  magic++;

  register int total_msg_size = 0;

  //first compute size
  for (i=0;i<NumPes;i++) {
    index = i;
    for (j=msgnum[index]-1; (j>=0); j--) {
      if (PeList[index][j]->magic != magic) {                
	total_msg_size += ALIGN8(PeList[index][j]->msgsize);
	total_msg_size += 2 * sizeof(int);
	PeList[index][j]->magic=magic;

	nmsgs ++;
      }            
    }
  }
    
  total_msg_size += ALIGN8(sizeof(AllToAllHdr));

  ComlibPrintf("[%d] Message Size %d, nmsgs %d **%d**\n", CkMyPe(), total_msg_size, nmsgs, sizeof(AllToAllHdr));
    
  //poiter to the combined message, UGLY NAME
  char *p = (char *) CmiAlloc(total_msg_size * sizeof(char));    

  ComlibPrintf("After cmialloc\n");

  //buffer to copy stuff into
  char *t = p; 
  char *junk = NULL;
    
  int refno = 0;

  AllToAllHdr ahdr;
  ahdr.refno = refno;
  ahdr.id = id;
  ahdr.size = *length;
  ahdr.ufield = ufield;
  ahdr.nmsgs = nmsgs;

  /*
    PACKINT(refno);    
    PACK(comID, id);
      
    PACKINT(ufield);
    PACKINT(nmsgs);
    //    PACKINT(dummy); //Aligning to 8 bytes
    */

  PACK(AllToAllHdr, ahdr);   

  int msg_offset = ALIGN8(sizeof(AllToAllHdr));
    
  //Increment magic again for creating the message
  magic++;
  for (i=0;i<NumPes;i++) {
    index=i;
    int ref = 0;
    int size;

    for (j=msgnum[index]-1; (j>=0); j--) {
      //Check if it is a duplicate
      if (PeList[index][j]->magic != magic) {                
	size = PeList[index][j]->msgsize;
	PACKMSG(&size, sizeof(int));
	PACKMSG(&ref, sizeof(int));
	PeList[index][j]->magic=magic;
	PACKMSG(PeList[index][j]->msg, size);

	msg_offset = ALIGN8(msg_offset);
      }

      //Free it when all the processors have gotten rid of it
      if (--(PeList[index][j]->refCount) <=0) {
	ComlibPrintf("before cmifree \n");
	CmiFree(PeList[index][j]->msg);   
	ComlibPrintf("after cmifree \n");

	PTFREE(PeList[index][j]);
      }
      //Assign the current processors message pointer to NULL
      PeList[index][j] = NULL;
    }
    msgnum[index] = 0;
  }
    
  *length = total_msg_size;
  return p;
}

#undef UNPACK
#define UNPACK(type,data) {junk=(char *)&(data); memcpy(junk, t, sizeof(type));t+=sizeof(type);}
#undef UNPACKMSG
#define UNPACKMSG(dest,src, size) { memcpy(dest, src, size); offset += size;}

int PeTable :: UnpackAndInsert(void *in) {
  char *junk;
  char *t =(char *)in + CmiReservedHeaderSize;
  int i, ufield, npe, pe, nummsgs, ptrlistindex=0;
  char *msgend, *msgcur;
  comID id;
  int refno = 0;
  int msgsize;

  register int offset;

  UNPACK(int, refno);
  ComlibPrintf("%d UnPackAndInsert\n", CkMyPe());
  UNPACK(comID, id);
  UNPACK(int, msgsize);
  UNPACK(int, ufield);
  UNPACK(int, nummsgs);
  UNPACK(int, npe);

  // unpack all messages into an array
  msgend = (char*)in + msgsize;
  msgcur = (char*)in + ALIGN8(sizeof(struct AllToAllHdr) + (npe+nummsgs+1)*sizeof(int));
  while (msgcur < msgend) {
    CmiChunkHeader *ch = (CmiChunkHeader *)msgcur;

    PTinfo *temp;
    PTALLOC(temp);
    temp->msgsize=ch->size;
    temp->msg=(void*)&ch[1];
    temp->refCount=0;
    temp->magic=0;
    temp->offset=0;

    ptrvec.insert(++ptrlistindex, temp);

    // fix the ref field of the message
    ch->ref = (int)((char*)in - (char*)temp->msg);
    msgcur += ALIGN8(ch->size) + sizeof(CmiChunkHeader);
  }

  pe = -1;
  //for (i=0;i<npe;i++) {
  for (i=0; i<nummsgs; ++i) {
    //UNPACK(int, pe);
    //pe *= -1;

    UNPACK(int, offset);
    if (offset <= 0) {
      pe = -1 * offset;
      --i;
      continue;
    }

    if (msgnum[pe] >= MaxSize[pe]) {
      REALLOC(PeList[pe], MaxSize[pe]);
      MaxSize[pe] *= 2;
    }
    PeList[pe][msgnum[pe]] = ptrvec[offset];
    (ptrvec[offset])->refCount++;
    msgnum[pe]++;

    /*
      while (offset > 0) {
      int tempmsgsize;
      UNPACKMSG(&(tempmsgsize), (char *)in+offset, sizeof(int));
      int ptr;
      UNPACKMSG(&ptr, (char *)in+offset, sizeof(int));

      if (ptr >=0 )  {
      if (msgnum[pe] >= MaxSize[pe]) {
      REALLOC(PeList[pe], MaxSize[pe]);
      MaxSize[pe] *= 2;
      }
      PeList[pe][msgnum[pe]]=ptrvec[ptr];
      (ptrvec[ptr])->refCount++;
      msgnum[pe]++;

      UNPACK(int, offset);
      continue;
      }
            
      PTinfo *temp;
      PTALLOC(temp);
      temp->msgsize=tempmsgsize;
      temp->refCount=1;
      temp->magic=0;
      temp->offset=0;

      ptrvec.insert(ptrlistindex, temp);
      memcpy((char *)in+offset-sizeof(int), &ptrlistindex, sizeof(int));

      ptrlistindex++;
      temp->msg=(void *)((char *)in+offset);
      if (msgnum[pe] >= MaxSize[pe]) {

      REALLOC(PeList[pe], MaxSize[pe]);
      MaxSize[pe] *= 2;
      }
      PeList[pe][msgnum[pe]]=temp;
      msgnum[pe]++;
      UNPACK(int, offset);
      //}
      //t -=sizeof(int);
      }
      *(int *)((char *)in -sizeof(int))=ptrlistindex; 
      */
  }

  REFFIELD(in) = ptrlistindex;
  if (ptrlistindex==0) {
    REFFIELD(in) = 1;
    CmiFree(in);
  }

  /*  
      for (i=0;i<ptrlistindex;i++) {
      char * actualmsg=(char *)(ptrvec[i]->msg);
      int *rc=(int *)(actualmsg-sizeof(int));
      *rc=(int)((char *)in-actualmsg);
      //ComlibPrintf("I am inserting %d\n", *rc);
      }
  */

  ptrvec.removeAll();
  
  return(ufield);
}

/* Unpack and insert an all to all message, the router provides the
   list of processors to insert into.
   Same protocol as mentioned earlier.
*/

int PeTable :: UnpackAndInsertAll(void *in, int npes, int *pelist) {
  char *junk;
  char *t =(char *)in /*+CmiReservedHeaderSize*/;
  int i,  
    ufield,   // user field or ths stage of the iteration 
    nmsgs,    // number of messages in combo message
    refno,    // reference number
    dummy;    // alignment dummy
  
  comID id;

  /*
    UNPACK(int, refno);      
    UNPACK(comID, id);
    
    UNPACK(int, ufield);
    UNPACK(int, nmsgs);
    //UNPACK(int, dummy);
    int header_size = sizeof(comID) + CmiReservedHeaderSize + 3 *sizeof(int);
    if(header_size % 8 != 0)
    t+= 8 - header_size % 8;
  */

  AllToAllHdr ahdr;
  UNPACK(AllToAllHdr, ahdr);

  if((sizeof(AllToAllHdr) & 7) != 0)
    t += 8 - (sizeof(AllToAllHdr) & 7);

  refno = ahdr.refno;
  id = ahdr.id;
  nmsgs = ahdr.nmsgs;
  ufield = ahdr.ufield;

  ComlibPrintf("[%d] unpack and insert all %d, %d\n", CkMyPe(), ufield, nmsgs);
  
  //Inserting a memory foot print may, change later
  CmiChunkHeader *chdr= (CmiChunkHeader*)((char*)in - sizeof(CmiChunkHeader));

  int *ref;
  int size;
  char *msg;
  for(int count = 0; count < nmsgs; count++){

    t += sizeof(CmiChunkHeader);
    msg = t;

    // Get the size of the message, and set the ref field correctly for CmiFree
    size = SIZEFIELD(msg);
    REFFIELD(msg) = (int)((char *)in - (char *)msg);

    t += ALIGN8(size);

    // Do CmiReference(msg), this is done bypassing converse!
    chdr->ref++;

    /*
      UNPACK(int, size);
      ref = (int *)t;
      t += sizeof(int);
    
      *ref = (int)((char *)(&chdr->ref) - (char *)ref);
      chdr->ref ++;

      ComlibPrintf("ref = %d, global_ref = %d\n", *ref, chdr->ref);
    
      msg = t;
      t += ALIGN8(size);
    */
    InsertMsgs(npes, pelist, size, msg);
  }  

  CmiFree(in);
  return ufield;
}

/*
 * Added by Filippo, 2005/04/18 to allow a zero-copy sending, through the
 * CmiVectorSend functions.
 */
PTvectorlist PeTable :: ExtractAndVectorize(comID id, int ufield, int npe, int *pelist) {
  char *junk;
  int nummsgs = 0;
  int offset;
  int i, j;

  for (i=0; i<npe; ++i) nummsgs += msgnum[pelist[i]];

  ComlibPrintf("%d In ExtractAndVectorize %d, %d\n", CmiMyPe(), npe, nummsgs); 

  if (nummsgs ==0) {
    ComlibPrintf("Returning NULL\n");
    return(NULL);
  }
    
  int headersize = sizeof(struct AllToAllHdr) + (npe + nummsgs + 1) * sizeof(int);
  //int msg_offset = CmiReservedHeaderSize + sizeof(comID) 
  //    + (npe + 4 + nummsgs) * sizeof(int);  

  char *p;
  p=(char *)CmiAlloc(headersize);

  char *t = p + CmiReservedHeaderSize;
  if (!p) CmiAbort("Big time problem\n");
  magic++;

  int refno = id.refno;    

  // SHOULD PACK THE SIZE, which is not available: fix elan and undo this cvs update
  PACK(int, refno);
  PACK(comID, id);
  PACK(int, ufield);
  PACK(int, nummsgs);
  PACK(int, npe);
    
  int lesspe=0;
  int npacked = 0;
  for (i=0;i<npe;i++) {
    int index=pelist[i];

    if (msgnum[index]<=0) {
      lesspe++;
            
      ComlibPrintf("[%d] msgnum[index]<=0 !!!!!\n", CkMyPe());
      continue;
    }
        
    ComlibPrintf("%d Packing pelist[%d]\n", CkMyPe(), index);
    register int newval=-1*pelist[i];
    PACK(int, newval); 

    for (j=0;j<msgnum[index];j++) {
      if (PeList[index][j]->magic == magic) {
	offset=(PeList[index][j]->offset);
      }
      else {
	npacked ++;
                
	//offset=msg_offset;
	offset=npacked;
	PeList[index][j]->magic=magic;
	PeList[index][j]->offset=offset;
	PTinfo *tempmsg=PeList[index][j];

	ptrvec.insert(npacked, tempmsg);
      }
      
      //ComlibPrintf("%d Packing msg_offset=%d\n", CkMyPe(), offset);
      PACK(int, offset); 

      --(PeList[index][j]->refCount);
      /*
	if (--(PeList[index][j]->refCount) <=0) {
	CmiFree(PeList[index][j]->msg);
                
	PTFREE(PeList[index][j]);
	}
      */
      PeList[index][j]=NULL;
    }
    msgnum[index]=0;
  }
  //offset=-1;
  //PACK(int, offset);

  // See petable.h for a description of this structure
  PTvectorlist result = (PTvectorlist)CmiAlloc(sizeof(struct ptvectorlist) +
					       (npacked+1)*2*sizeof(char*) +
					       2*sizeof(CmiChunkHeader));
  result->count = npacked + 1;
  result->sizes = (int*)((char*)result + sizeof(struct ptvectorlist) + sizeof(CmiChunkHeader));
  result->msgs  = (char**)((char*)result->sizes + (npacked+1)*sizeof(int) + sizeof(CmiChunkHeader));

  SIZEFIELD(result->sizes) = (npacked+1)*sizeof(int);
  REFFIELD(result->sizes) = - (int)(sizeof(struct ptvectorlist) + sizeof(CmiChunkHeader));
  SIZEFIELD(result->msgs) = (npacked+1)*sizeof(int);
  REFFIELD(result->msgs) = - (sizeof(struct ptvectorlist) + (npacked+1)*sizeof(int) + 2*sizeof(CmiChunkHeader));
  CmiReference(result);

  result->sizes[0] = headersize;
  result->msgs[0] = p;
  PTinfo *temp;
  for (i=1; i<=npacked; ++i) {
    temp = ptrvec[i];
    result->sizes[i] = temp->msgsize;
    result->msgs[i] = (char*)temp->msg;
    // if there is still reference we CmiReference the message so it does not get deleted
    // otherwise we free also the PTinfo struct use to hold it
    if (temp->refCount > 0) CmiReference(result->msgs[i]);
    else PTFREE(temp);
  }

  ptrvec.removeAll();

  /*    
	if (lesspe) {
	t=p+CmiReservedHeaderSize+2*sizeof(int) + sizeof(comID);
	npe=npe-lesspe;
	PACK(int, npe);
	}
  */

  //return(p);

  return result;
}

/*
 * Added by Filippo, 2005/04/18 to allow a zero-copy sending, through the
 * CmiVectorSend functions.
 */
PTvectorlist PeTable :: ExtractAndVectorizeAll(comID id, int ufield) {
  int nmsgs = 0, i, j;
  int index = 0;

  ComlibPrintf("[%d] In Extract And Vectorize All\n", CkMyPe());

  //Increment magic to detect duplicate messages
  magic++;

  //register int total_msg_size = 0;

  //first compute size
  for (i=0;i<NumPes;i++) {
    index = i;
    for (j=msgnum[index]-1; (j>=0); j--) {
      if (PeList[index][j]->magic != magic) {                
	//total_msg_size += ALIGN8(PeList[index][j]->msgsize);
	//total_msg_size += 2 * sizeof(int);
	PeList[index][j]->magic=magic;
	ptrvec.insert(nmsgs, PeList[index][j]);
	PeList[index][j] = NULL;
	nmsgs ++;
      }
      --(PeList[index][j]->refCount);
    }
  }

  //total_msg_size += ALIGN8(sizeof(AllToAllHdr));

  ComlibPrintf("[%d] nmsgs %d **%d**\n", CkMyPe(), nmsgs, sizeof(AllToAllHdr));
    
  //poiter to the message header
  AllToAllHdr *ahdr = (AllToAllHdr *) CmiAlloc(sizeof(struct AllToAllHdr));

  ComlibPrintf("After cmialloc\n");

  /*
  //buffer to copy stuff into
  char *t = p; 
  char *junk = NULL;
    
  int dummy = 0;
  */

  int refno = 0;

  ahdr->refno = refno;
  ahdr->id = id;
  ahdr->ufield = ufield;
  ahdr->nmsgs = nmsgs;

  /*
    PACKINT(refno);    
    PACK(comID, id);
      
    PACKINT(ufield);
    PACKINT(nmsgs);
    //    PACKINT(dummy); //Aligning to 8 bytes
    */

  // See petable.h for a description of this structure
  PTvectorlist result = (PTvectorlist)CmiAlloc(sizeof(struct ptvectorlist) +
					       (nmsgs+1)*2*sizeof(char*) +
					       2*sizeof(CmiChunkHeader));
  result->count = nmsgs + 1;
  result->sizes = (int*)((char*)result + sizeof(struct ptvectorlist) + sizeof(CmiChunkHeader));
  result->msgs  = (char**)((char*)result->sizes + (nmsgs+1)*sizeof(int) + sizeof(CmiChunkHeader));

  SIZEFIELD(result->sizes) = (nmsgs+1)*sizeof(int);
  REFFIELD(result->sizes) = - (int)(sizeof(struct ptvectorlist) + sizeof(CmiChunkHeader));
  SIZEFIELD(result->msgs) = (nmsgs+1)*sizeof(int);
  REFFIELD(result->msgs) = - (sizeof(struct ptvectorlist) + (nmsgs+1)*sizeof(int) + 2*sizeof(CmiChunkHeader));
  CmiReference(result);

  result->sizes[0] = sizeof(ahdr);
  result->msgs[0] = (char*)ahdr;
  PTinfo *temp;
  for (i=1; i<nmsgs; ++i) {
    temp = ptrvec[i];
    result->sizes[i] = temp->msgsize;
    result->msgs[i] = (char*)temp->msg;
    // if there is still reference we CmiReference the message so it does not get deleted
    // otherwise we free also the PTinfo struct use to hold it
    if (temp->refCount > 0) CmiReference(result->msgs[i]);
    else PTFREE(temp);
  }

  ptrvec.removeAll();

  return result;

  /*
  PACK(AllToAllHdr, ahdr);   

  int msg_offset = ALIGN8(sizeof(AllToAllHdr));
    
  //Increment magic again for creating the message
  magic++;
  for (i=0;i<NumPes;i++) {
    index=i;
    int ref = 0;
    int size;

    for (j=msgnum[index]-1; (j>=0); j--) {
      //Check if it is a duplicate
      if (PeList[index][j]->magic != magic) {                
	size = PeList[index][j]->msgsize;
	PACKMSG(&size, sizeof(int));
	PACKMSG(&ref, sizeof(int));
	PeList[index][j]->magic=magic;
	PACKMSG(PeList[index][j]->msg, size);

	msg_offset = ALIGN8(msg_offset);
      }

      //Free it when all the processors have gotten rid of it
      if (--(PeList[index][j]->refCount) <=0) {
	ComlibPrintf("before cmifree \n");
	CmiFree(PeList[index][j]->msg);   
	ComlibPrintf("after cmifree \n");

	PTFREE(PeList[index][j]);
      }
      //Assign the current processors message pointer to NULL
      PeList[index][j] = NULL;
    }
    msgnum[index] = 0;
  }
    
  *length = total_msg_size;
  return p;
  */
}

void PeTable :: GarbageCollect() {
}

/*@}*/

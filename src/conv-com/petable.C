/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/*********************************************
 * File : petable.C
 *
 * Author: Krishnan V
 *
 * The message buffer
 *********************************************/
#include <string.h>
#include <stdlib.h>
#include <converse.h>
#include "comlib.h"
#include "petable.h"
#include "converse.h"

#define BIGBUFFERSIZE 65536
#define PTPREALLOC    100

struct AllToAllHdr{
    char dummy[CmiReservedHeaderSize];
    int refno;
    comID id;
    int ufield;
    int nmsgs;
};


/**************************************************************
 * Preallocated memory=P*MSGQLEN ptr + 2P ints + 1000 ptrs
 **************************************************************/
PeTable :: PeTable(int n)
{
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
}

PeTable :: ~PeTable()
{
  int i;
  for (i=0;i<NumPes;i++) CmiFree(PeList[i]);
  CmiFree(PeList);
  delete msgnum;
  delete MaxSize;
  GarbageCollect();
  //CmiFree(ptrlist);
  PTinfo *tmp;
  while (PTFreeList) {
  	tmp=PTFreeList;
	PTFreeList=tmp->next;
	CmiFree(tmp);
  }
 // delete FreeList;

}

void PeTable:: Purge()
{
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

void PeTable :: ExtractAndDeliverLocalMsgs(int index)
{
  int j;
  msgstruct m;

  ComlibPrintf("%d:Delivering %d local messages\n", CkMyPe(), msgnum[index]);
  for (j=msgnum[index]-1;(j>=0);j--) {

	m.msgsize=PeList[index][j]->msgsize;
	m.msg=PeList[index][j]->msg;

	if (--(PeList[index][j]->refCount) <=0) {
            CmiSyncSendAndFree(CkMyPe()/*index*/, m.msgsize, (char*)m.msg);
            PTFREE(PeList[index][j]);
	}
	else {
            CmiSyncSend(CkMyPe()/*index*/, m.msgsize, (char*)m.msg);
        }
	PeList[index][j]=NULL;
  }
  msgnum[index]=j+1;

  return;
}


#undef PACK
#undef PACKMSG
#define PACKINT(data) {((int*)t)[0] = data; t+=sizeof(int);}
#define PACK(type,data) {junk=(char *)&(data); memcpy(t, junk, sizeof(type)); t+=sizeof(type);}
#define PACKMSG(data, size) { memcpy(p+msg_offset, (data), size); msg_offset += size; }

/*Used for all to all multicast operations.  Assumes that each message
  is destined to all the processors, to speed up all to all
  substantially --Sameer 09/03/03 
  
  Protocol:
  |ref|comid|ufield|nmsgs|size|ref|msg1|size2|ref2|msg2|....
*/

char * PeTable ::ExtractAndPackAll(comID id, int ufield, int *length)
{
    int nmsgs = 0, i, j;
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
    
    int dummy = 0;
    
    int refno = 0;

    AllToAllHdr ahdr;
    ahdr.refno = refno;
    ahdr.id = id;
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

char * PeTable ::ExtractAndPack(comID id, int ufield, int npe, 
                                int *pelist, int *length)
{
    char *junk;
    int mask=~7;
    int nummsgs, offset, actual_msgsize=0, num_distinctmsgs;
    
    ComlibPrintf("In ExtractAndPack %d\n", npe); 
    
    int tot_msgsize=TotalMsgSize(npe, pelist, &nummsgs, &num_distinctmsgs);
    if (tot_msgsize ==0) {
	*length=0;
        
        ComlibPrintf("Returning NULL\n");

	return(NULL);
    }
    
    //int ave_msgsize=(tot_msgsize>MSGSIZETHRESHOLD) ? 
    //  tot_msgsize/(num_distinctmsgs):tot_msgsize;
    
    int msg_offset = CmiReservedHeaderSize + sizeof(comID) 
        + (npe + 4 + nummsgs) * sizeof(int);  

    msg_offset = ALIGN8(msg_offset);
    
    int headersize=msg_offset;
    
    *length = tot_msgsize;
    *length += msg_offset;
    char *p;
    p=(char *)CmiAlloc(*length);
    
    int l1 = *length;
    
    char *t = p + CmiReservedHeaderSize;
    int i, j;
    if (!p) CmiAbort("Big time problem\n");
    magic++;

    int refno = id.refno;    

    PACKINT(refno);
    PACK(comID, id);
    PACKINT(ufield);
    PACKINT(npe);
    
    int lesspe=0;
    int npacked = 0;
    for (i=0;i<npe;i++) {
        int index=pelist[i];

        if (msgnum[index]<=0) {
            lesspe++;
            continue;
        }
        
        //ComlibPrintf("%d Packing pelist[%d]\n", CkMyPe(), i);
        register int newval=-1*pelist[i];
        PACKINT(newval); 
        for (j=0;j<msgnum[index];j++) {
            if (PeList[index][j]->magic != magic) {
		int tmpms=actual_msgsize+PeList[index][j]->msgsize;
		if (tmpms >= MSGSIZETHRESHOLD 
                    /*|| (PeList[index][j]->msgsize>=ave_msgsize)*/ ) {
                    
                    CmiPrintf("%d sending directly\n", CkMyPe());
                    if (--(PeList[index][j]->refCount) <=0) {
                        CmiSyncSendAndFree(index, PeList[index][j]->msgsize, 
                                           (char*)PeList[index][j]->msg);
			//ComlibPrintf("%d Freeing msg\n", CkMyPe());
                        PTFREE(PeList[index][j]);
			}
                    else
                        CmiSyncSend(index, PeList[index][j]->msgsize, 
                                    (char*)PeList[index][j]->msg);
                    PeList[index][j]=NULL;
                    continue;
		}
		
		npacked ++;
                
     		offset=msg_offset;
		PeList[index][j]->magic=magic;
		PeList[index][j]->offset=msg_offset;
		PTinfo *tempmsg=PeList[index][j];
 		PACKMSG((&(tempmsg->msgsize)), sizeof(int));
		actual_msgsize += tempmsg->msgsize;
		int nullptr=-1;
 		PACKMSG(&nullptr, sizeof(int));

     		PACKMSG(tempmsg->msg, tempmsg->msgsize);

                msg_offset = ALIGN8(msg_offset);
                actual_msgsize = ALIGN8(actual_msgsize);                
		actual_msgsize += 2*sizeof(int);
            }
            else {
		offset=(PeList[index][j]->offset);
            }
            
            //ComlibPrintf("%d Packing msg_offset=%d\n", CkMyPe(), offset);
            PACKINT(offset); 
            if (--(PeList[index][j]->refCount) <=0) {
                CmiFree(PeList[index][j]->msg);
                
                PTFREE(PeList[index][j]);
            }
            PeList[index][j]=NULL;
        }
        msgnum[index]=0;
    }
    offset=-1;
    PACKINT(offset);
    
    if (lesspe) {
        t=p+CmiReservedHeaderSize+2*sizeof(int) + sizeof(comID);
	npe=npe-lesspe;
	PACK(int, npe);
    }
    if (!actual_msgsize) {
        if (l1 >= MAXBUFSIZE) 
            CmiFree(p);
        *length=0;
        return(NULL);
    }
    
    *length=actual_msgsize+headersize;
    return(p);
} 

#undef UNPACK
#define UNPACK(type,data) {junk=(char *)&(data); memcpy(junk, t, sizeof(type));t+=sizeof(type);}
#undef UNPACKMSG
#define UNPACKMSG(dest,src, size) { memcpy(dest, src, size); offset += size;}

int PeTable :: UnpackAndInsert(void *in)
{
  char *junk;
  char *t =(char *)in + CmiReservedHeaderSize;
  int i, ufield, npe, pe, tot_msgsize, ptrlistindex=0;
  comID id;
  int refno = 0;

  UNPACK(int, refno);
  
  //ComlibPrintf("%d UnPacking id\n", CkMyPe());
  UNPACK(comID, id);
  UNPACK(int, ufield);
  UNPACK(int, npe);
  
  register int offset;
  for (i=0;i<npe;i++) {
	UNPACK(int, pe);
	pe *= -1;

	UNPACK(int, offset);
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
	}
	t -=sizeof(int);
  }
  *(int *)((char *)in -sizeof(int))=ptrlistindex; 
  
  if (ptrlistindex==0)
      CmiFree(in);
  
  for (i=0;i<ptrlistindex;i++) {
      char * actualmsg=(char *)(ptrvec[i]->msg);
      int *rc=(int *)(actualmsg-sizeof(int));
      *rc=(int)((char *)in-actualmsg);
      //ComlibPrintf("I am inserting %d\n", *rc);
  }
  
  return(ufield);
}

/* Unpack and insert an all to all message, the router provides the
   list of processors to insert into.
   Same protocol as mentioned earlier.
*/

int PeTable :: UnpackAndInsertAll(void *in, int npes, int *pelist){
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

  if(sizeof(AllToAllHdr) % 8 != 0)
      t += 8 - sizeof(AllToAllHdr) % 8;

  refno = ahdr.refno;
  id = ahdr.id;
  nmsgs = ahdr.nmsgs;
  ufield = ahdr.ufield;

  ComlibPrintf("[%d] unpack and insert all %d, %d\n", CkMyPe(), ufield, nmsgs);
  
  //Inserting a memory foot print may, change later
  CmiChunkHeader *chdr= (CmiChunkHeader*)((char*)in - sizeof(CmiChunkHeader));

  for(int count = 0; count < nmsgs; count++){
      int *ref = 0;
      int size = 0;
      char *msg = 0;

      UNPACK(int, size);
      ref = (int *)t;
      t += sizeof(int);

      *ref = (int)((char *)(&chdr->ref) - (char *)ref);
      chdr->ref ++;

      ComlibPrintf("ref = %d, global_ref = %d\n", *ref, chdr->ref);

      msg = t;
      t += ALIGN8(size);
      
      InsertMsgs(npes, pelist, size, msg);
  }  

  CmiFree(in);
  return ufield;
}

void PeTable :: GarbageCollect()
{
}

GList :: GList()
{
  InList=(InNode *)CmiAlloc(10*sizeof(InNode));
  InListIndex=0;
}

GList :: ~GList()
{
  CmiFree(InList);
}

int GList :: AddWholeMsg(void *ptr)
{
  InList[InListIndex].flag=0;
  InList[InListIndex++].ptr=ptr;
  return(InListIndex-1);
}

void GList :: setRefcount(int indx, int ref)
{
  //ComlibPrintf("setting InList[%d]=%d\n", indx, ref);
  InList[indx].refCount=ref;
}

void GList :: DeleteWholeMsg(int indx)
{
  //ComlibPrintf("DeleteWholeMsg indx=%d\n", indx);
  InList[indx].refCount--;
  if ((InList[indx].refCount <=0) && (InList[indx].flag==0)) {
	//ComlibPrintf("Deleting msgwhole\n");
	CmiFree(InList[indx].ptr);
  }
}
void GList :: DeleteWholeMsg(int indx, int flag)
{
  InList[indx].refCount--;
  InList[indx].flag=flag;
} 
/*
void GList :: DeleteWholeMsg(int indx, void *p)
{
  int *rc=(int *)((char *)p-sizeof(int));
  *rc=(int)((char *)p-(char *)(InList[indx].ptr)+2*sizeof(int));
}
*/

void GList :: GarbageCollect()
{
  for (int i=0;i<InListIndex;i++) {
	if ((InList[i].refCount <= 0) && (InList[i].flag >0))
		CmiFree(InList[i].ptr);
  }
}

void GList :: Add(void *ptr )
{
  InList[InListIndex].ptr=ptr;
  InList[InListIndex++].refCount=0;
}

void GList :: Delete()
{
  InListIndex=0;
  /******
  int counter=0;
  for (int i=0;i< InListIndex;i++) {
  	if (InList[i].refCount <=0)  {
		counter++;
		CmiFree(InList[i].ptr);
	}
  }
  if (counter == InListIndex) InListIndex=0;
****/
}


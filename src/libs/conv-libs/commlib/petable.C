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
#include "commlib.h"
#include "petable.h"

#define BIGBUFFERSIZE 10240
#define PTPREALLOC    100

int KMyActiveRefno(comID);

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

char CombBuffer[128][MAXBUFSIZE];
int combcount;

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

  ptrlist=(PTinfo **)CmiAlloc(1000*sizeof(PTinfo *));
  //  FreeList= new GList;
  //CombBuffer=(char *)CmiAlloc(BIGBUFFERSIZE);

  combcount = 0;
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
  CmiFree(ptrlist);
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
            // ComlibPrintf("%d Warning: %d Undelivered Messages for %d\n", CmiMyPe(), msgnum[i], i);
	  //msgnum[i]=0;
	}
  }
  GarbageCollect();
  //  ComlibPrintf("combcount = %d\n", combcount);
  //combcount = 0;
}

void PeTable :: InsertMsgs(int npe, int *pelist, int size, void *msg)
{
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
    
    //    ComlibPrintf("Inserting %d %d %d\n", msgnum[index], CmiMyPe(), index);
    
    if (msgnum[index] >= MaxSize[index]) {
      ComlibPrintf("reallocing... %d %d %d\n", msgnum[index], CmiMyPe(), index);
      //PeList[index]=(PTinfo **)realloc(PeList[index], 2*sizeof(PTinfo *)*MaxSize[index]);
      REALLOC(PeList[index], MaxSize[index]);
      MaxSize[index] *= 2;
    }
    PeList[index][msgnum[index]]=tmp;
    msgnum[index]++;
  }
}

void PeTable :: InsertMsgs(int npe, int *pelist, int nmsgs, void **msglist)
{
  msgstruct **m=(msgstruct **)msglist;
  for (int i=0;i<nmsgs;i++)
  	InsertMsgs(npe, pelist, m[i]->msgsize, m[i]->msg);
}

int PeTable :: ExtractMsgs(int npe, int *pelist, int *nmsgs, void **msgs)
{
  int nm=0, i, j;
  msgstruct *m;
  magic++;

  //ComlibPrintf("%d Extract called\n",CmiMyPe());

  for (i=0;i<npe;i++) {
     int index=pelist[i];
     for (j=msgnum[index]-1;(j>=0) && (nm < MAXNUMMSGS) ;j--) {
	if (PeList[index][j]->magic != magic) {
	  m=(msgstruct *)msgs[nm++];
	  m->msgsize=PeList[index][j]->msgsize;
	  PeList[index][j]->magic=magic;
	  if (--(PeList[index][j]->refCount) <=0) {
		m->msg=PeList[index][j]->msg;
	  	//CmiFree(PeList[index][j]);
	  	PTFREE(PeList[index][j]);
	  }
	  else {
		m->msg=CmiAlloc(m->msgsize);
		memcpy(m->msg, PeList[index][j]->msg, m->msgsize); 
  	  }
	}
	PeList[index][j]=NULL;
     }
     msgnum[index]=j+1;
  }

  *nmsgs=nm;
  return(i);
}
	
void PeTable :: ExtractAndDeliverLocalMsgs(int index)
{
  int j;
  msgstruct m;

  ComlibPrintf("%d:Delivering %d local messages\n", CmiMyPe(), msgnum[index]);
  for (j=msgnum[index]-1;(j>=0);j--) {

	m.msgsize=PeList[index][j]->msgsize;
	m.msg=PeList[index][j]->msg;

	if (--(PeList[index][j]->refCount) <=0) {
           CmiSyncSendAndFree(CmiMyPe()/*index*/, m.msgsize, m.msg);
	   PTFREE(PeList[index][j]);
	  //CmiFree(PeList[index][j]);
	}
	else {
           CmiSyncSend(CmiMyPe()/*index*/, m.msgsize, m.msg);
        }
	PeList[index][j]=NULL;
  }
  msgnum[index]=j+1;

  return;
}

int PeTable :: TotalMsgSize(int npe, int *pelist, int *nm, int *nd)
{
  int totsize=0, mask=~7;
  int perpesize=0;
  magic++;
  *nm=0;
  *nd=0;

  //  ComlibPrintf("%d Extract called %d %d\n",CmiMyPe(), npe, pelist[npe-1]);

  for (int i=0;i<npe;i++) {
   
      //    ComlibPrintf("\nIn Loop\n");
 
    int index=pelist[i];
    
    if (index > NumPes) {
      ComlibPrintf("index=%d > NumPes=%d\n", index, NumPes);
      return(0);
    }
    *nm += msgnum[index];
    perpesize=0;
    for (int j=0;j<msgnum[index];j++) {
      if (PeList[index][j]->magic != magic) {
          //        ComlibPrintf("Computing totsize\n");
        totsize += (PeList[index][j]->msgsize+7)&mask;
        totsize += sizeof(int)+sizeof(int);
        perpesize += (PeList[index][j]->msgsize+7)&mask;
        perpesize += sizeof(int)+sizeof(int);
        PeList[index][j]->magic=magic;
        (*nd)++;
      }
    }
    //if (ALPHA*msgnum[index] < BETA*perpesize) {
    //	totsize -= perpesize;
    //	SendDirect(index, perpesize);
    //    }
  }
  return(totsize);
}


comID defid;
  
#undef PACK
#define PACK(type,data) {junk=(char *)&(data); for(int i=0;i< sizeof(type);i++) t[i]=junk[i];t+=sizeof(type);}
#undef PACKMSG
#define PACKMSG(data, size) { memcpy(p+msg_offset, (data), size); msg_offset += size; }

char * PeTable ::ExtractAndPack(comID id, int ufield, int npe, int *pelist, int *length)
{
  char *junk;
  int mask=~7;
  int nummsgs, offset, actual_msgsize=0, num_distinctmsgs;

  //  ComlibPrintf("In ExtractAndPack %d\n", npe); 

  int tot_msgsize=TotalMsgSize(npe, pelist, &nummsgs, &num_distinctmsgs);
  if (tot_msgsize ==0) {
	*length=0;
	return(NULL);
  }

  int ave_msgsize=(tot_msgsize>MSGSIZETHRESHOLD) ? tot_msgsize/(num_distinctmsgs):tot_msgsize;
  int msg_offset= (CmiMsgHeaderSizeBytes+sizeof(comID)+ (npe+4+nummsgs)*sizeof(int)+7)&mask;  
  //int msg_offset= (CmiMsgHeaderSizeBytes+/*sizeof(comID)*/+ (npe+4+nummsgs)*sizeof(int)+7)&mask;  

  int headersize=msg_offset;
  //*length=tot_msgsize+ msg_offset;
  *length=(tot_msgsize>MSGSIZETHRESHOLD) ? MSGSIZETHRESHOLD : tot_msgsize;
  *length += msg_offset;
  char *p;

  if(combcount == 128)
    combcount = 0;
  if ((*length) < MAXBUFSIZE) {
    p=CombBuffer[combcount];
    combcount ++;
  }
  else 
    p=(char *)CmiAlloc(*length);

  int l1 = *length;
  
  //  ComlibPrintf("%d header=%d total=%d\n", CmiMyPe(), headersize, *length);
  //  p=(char *)CmiAlloc(*length);
  char *t=p+CmiMsgHeaderSizeBytes;
  int i, j;
  if (!p) ComlibPrintf("Big time problem\n");
  magic++;
  //ComlibPrintf("%d Packing tot_msgsize\n", CmiMyPe());
  int refno=KMyActiveRefno(id);
  PACK(int, refno);
  //ComlibPrintf("%d Packing comID\n", CmiMyPe());

  //defid = id;
  PACK(comID, id);

  PACK(int, ufield);
  //ComlibPrintf("%d Packing pelistsize\n", CmiMyPe());
  PACK(int, npe);
  int lesspe=0;

  int npacked = 0;
  for (i=0;i<npe;i++) {
     int index=pelist[i];
     if (msgnum[index]<=0) {
	lesspe++;
	continue;
     }
  //ComlibPrintf("%d Packing pelist[%d]\n", CmiMyPe(), i);
     int newval=-1*pelist[i];
     PACK(int, newval); 
     for (j=0;j<msgnum[index];j++) {
	if (PeList[index][j]->magic != magic) {
		int tmpms=actual_msgsize+PeList[index][j]->msgsize;
		if (tmpms >= MSGSIZETHRESHOLD || (PeList[index][j]->msgsize>=ave_msgsize) ) {
		  //ComlibPrintf("%d sending directly\n", CmiMyPe());
			if (--(PeList[index][j]->refCount) <=0) {
				CmiSyncSendAndFree(index, PeList[index][j]->msgsize, PeList[index][j]->msg);
				//ComlibPrintf("%d Freeing msg\n", CmiMyPe());
	  			PTFREE(PeList[index][j]);
			}
			else
				CmiSyncSend(index, PeList[index][j]->msgsize, PeList[index][j]->msg);
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
                ComlibPrintf("%d Packing m[%d]->msg of size=%d\n", CmiMyPe(), i, tempmsg->msgsize);
     		PACKMSG(tempmsg->msg, tempmsg->msgsize);
		msg_offset = (msg_offset+7)&mask;
		actual_msgsize= (actual_msgsize+7)&mask;
		actual_msgsize+=2*sizeof(int);
	}
	else {
		offset=(PeList[index][j]->offset);
	}
  //ComlibPrintf("%d Packing msg_offset=%d\n", CmiMyPe(), offset);
     	PACK(int, offset); 
	if (--(PeList[index][j]->refCount) <=0) {
		CmiFree(PeList[index][j]->msg);
	  	//CmiFree(PeList[index][j]);
	  	PTFREE(PeList[index][j]);
	}
	PeList[index][j]=NULL;
      }
      msgnum[index]=0;
  //ComlibPrintf("%d Done Packing pesize=%d\n", CmiMyPe(), pelistsize);
  }
  offset=-1;
  PACK(int, offset);

  if (lesspe) {
        t=p+CmiMsgHeaderSizeBytes+2*sizeof(int) + sizeof(comID);
	npe=npe-lesspe;
	PACK(int, npe);
  }
  if (!actual_msgsize) {
	CmiFree(p);
	*length=0;
	return(NULL);
  }

  //  ComlibPrintf("Out of extract and pack in %d %d\n", CmiMyPe(), npacked);

  *length=actual_msgsize+headersize;
  //  ComlibPrintf("actual=%d, len=%d , %d\n", actual_msgsize+headersize, *length, nummsgs);
#if 0 /*Sameer: what the heck is this?  It only compiles under mpi- versions */
  if (l1 < MAXBUFSIZE) 
      ((CmiMsgHeaderBasic *)p)->rank = 1000;
#endif
  return(p);
} 

#undef UNPACK
#define UNPACK(type,data) {junk=(char *)&(data); for(int i=0;i< sizeof(type);i++) junk[i]=t[i];t+=sizeof(type);}
#undef UNPACKMSG
#define UNPACKMSG(dest,src, size) { memcpy(dest, src, size); offset += size;}

int PeTable :: UnpackAndInsert(void *in)
{
  char *junk;
  char *t =(char *)in + CmiMsgHeaderSizeBytes;
  int i, ufield, npe, pe, tot_msgsize, ptrlistindex=0;
  comID id;

  //  ComlibPrintf("In Unpack and Insert at %d\n", CmiMyPe());
  //PTinfo **ptrlist=(PTinfo **)CmiAlloc(100*sizeof(PTinfo *));

  //ComlibPrintf("%d UnPacking tot_msgsize\n", CmiMyPe());
  UNPACK(int, tot_msgsize);

  //ComlibPrintf("%d UnPacking id\n", CmiMyPe());
  UNPACK(comID, id);
  //  id = defid;

  UNPACK(int, ufield);
  UNPACK(int, npe);
  //  ComlibPrintf("%d UnPacking npe=%d\n", CmiMyPe(), npe);
  int offset;
  for (i=0;i<npe;i++) {
	UNPACK(int, pe);
	pe *= -1;
  //ComlibPrintf("%d UnPacking pelist[%d]=%d\n", CmiMyPe(), i, pe);
	UNPACK(int, offset);
  //ComlibPrintf("%d UnPacking offset= %d\n", CmiMyPe(), offset);
	while (offset > 0) {
                //ComlibPrintf("%d PeList[%d][%d]processed\n", CmiMyPe(), pe, msgnum[pe]);
		int tempmsgsize;
 		UNPACKMSG(&(tempmsgsize), (char *)in+offset, sizeof(int));
                //        ComlibPrintf("%d UnPacking m[%d]->msgsize=%d\n", CmiMyPe(), i, tempmsgsize);
		int ptr;
		UNPACKMSG(&ptr, (char *)in+offset, sizeof(int));
  //ComlibPrintf("%d ptr=%d\n", CmiMyPe(), ptr);
		if (ptr >=0 )  {
			if (msgnum[pe] >= MaxSize[pe]) {
			//	PeList[pe]=(PTinfo **)realloc(PeList[pe], 2*sizeof(PTinfo *)*MaxSize[pe]);
				REALLOC(PeList[pe], MaxSize[pe]);
				MaxSize[pe] *= 2;
			}
  			PeList[pe][msgnum[pe]]=ptrlist[ptr];
			(ptrlist[ptr])->refCount++;
			msgnum[pe]++;
			//ComlibPrintf("Should go back to loop\n");
			UNPACK(int, offset);
  //ComlibPrintf("%d UnPacking offset= %d\n", CmiMyPe(), offset);
			continue;
		}
  		//PTinfo * temp=(PTinfo *)CmiAlloc(sizeof(PTinfo));
		PTinfo *temp;
		PTALLOC(temp);
		temp->msgsize=tempmsgsize;
		temp->refCount=1;
		temp->magic=0;
		temp->offset=0;
		//ComlibPrintf("%d Unpack freelistindex=%d\n", CmiMyPe(), freelistindex);
		ptrlist[ptrlistindex]=temp;
		memcpy((char *)in+offset-sizeof(int), &ptrlistindex, sizeof(int));
		//ComlibPrintf("%d storing index=%d\n", CmiMyPe(), ptrlistindex);
		ptrlistindex++;
		//ComlibPrintf("Stroing msg from offset=%d\n", offset);
		temp->msg=(void *)((char *)in+offset);
		if (msgnum[pe] >= MaxSize[pe]) {
			//void **tmpptr=(void **)realloc(PeList[pe], 2*sizeof(PTinfo *)*MaxSize[pe]);
			//PeList[pe]=(PTinfo **)tmpptr;
			REALLOC(PeList[pe], MaxSize[pe]);
			MaxSize[pe] *= 2;
		}
  		PeList[pe][msgnum[pe]]=temp;
		msgnum[pe]++;
		UNPACK(int, offset);
  //ComlibPrintf("%d UnPacking offset= %d\n", CmiMyPe(), offset);
	}
	t -=sizeof(int);
  }
  *(int *)((char *)in -sizeof(int))=ptrlistindex; 
  if (ptrlistindex==0) CmiFree(in);
  for (i=0;i<ptrlistindex;i++) {
	char * actualmsg=(char *)(ptrlist[i]->msg);
	int *rc=(int *)(actualmsg-sizeof(int));
	*rc=(int)((char *)in-actualmsg);
	//ComlibPrintf("I am inserting %d\n", *rc);
  }
  //CmiFree(ptrlist);
//  ComlibPrintf("%d Unpack done pesize=%d\n", CmiMyPe(), pelistsize);
  return(ufield);
}

/******************************
  Trying to use Vector Send


int PeTable ::ExtractAsVector(comID id, int ufield, int npe, int *pelist, int **lengths, char ***msgvect)
{
  int nummsgs, offset, len;
  int tot_msgsize=TotalMsgSize(npe, pelist, &nummsgs);
  if (tot_msgsize ==0) {
	lengths=NULL;
	return(0);
  }
  len=2+2*nummsgs;
  char **mvector=(char **)CmiAlloc(len*sizeof(char *));
  int *sizes=(int *)CmiAlloc(len*sizeof(int));
  int msg_offset= sizeof(HEADMSG)+ (npe+1+nummsgs)*sizeof(int);  
  int i, j;
  magic++

  HEADMSG *headermsg=(HEADMSG *)CmiAlloc(sizeof(HEADMSG));
  headermsg->tot_msgsize==KMyActiveRefno(id);
  headermsg->id=id;
  headermsg->ufield=ufield;
  headermsg->npe=npe;
  sizes[0]=sizeof(HEADMSG);
  sizes[1]=((sizeof(HEADMSG)+7)&mask)-sizes[0]+2*sizeof(int);
  mvector[0]=(char *)headermsg;
  mvector[1]=pad;

  sizes[2]=(npe+1+nummsgs)*sizeof(int);
  sizes[3]=((npe+1+nummsgs)*sizeof(int)+7)&mask)-sizes[2]+2*sizeof(int);
  mvector[2]=(char *)CmiAlloc(sizes[1]);
  mvector[3]=pad;
  int *pehead=(int *)mvector[2];
  int peheadindex=0;

  int mvectindex=npe+2;
  int *mhead=(int *)CmiAlloc(2*nummsgs*sizeof(int));
  int mheadindex=0;

  int msg_offset=sizes[0]+sizes[1]+sizes[2]+sizes[3];
  int mvectorindex=4;
  for (i=0;i<npe;i++) {
     int index=pelist[i];
     pehead[peheadindex++]= -1*index;

     for (j=0;j<msgnum[index];j++) {
	if (PeList[index][j]->magic != magic) {
     		offset=msg_offset;
		PeList[index][j]->magic=magic;
		PeList[index][j]->offset=msg_offset;
		msgstruct *tempmsg=(msgstruct *)(PeList[index][j]->msg);

		sizes[mvectorindex]=2*sizeof(int);
		mvector[mvectorindex++]=(char *)(mhead+mheadindex);
 		mhead[mheadindex++]=tempmsg->msgsize;
 		mhead[mheadindex++]=-1;
		sizes[mvectorindex]=((2*sizeof(int)+7)&mask);
		mvector[mvectorindex++]=pad;
		sizes[mvectorindex]=tempmsg->msgsize;
		mvector[mvectorindex++]=(char *)(tempmsg->msg);
		sizes[mvectorindex]=((tempmsg->msgsize + 7)&mask)-(tempmsg->msgsize)+2*sizeof(int);
		mvector[mvectorindex++]=pad;
		for (int ll=1;ll<5;ll++)
			msg_offset += sizes[mvectorindex-ll];
	}
	else {
		offset=PeList[index][j]->offset;
	}
  	//ComlibPrintf("%d packing offset=%d\n",CmiMyPe(), offset);
        pehead[peheadindex++]=offset;
	if (--(PeList[index][j]->refCount) <=0) {
		//ComlibPrintf("freeing pelist[%d][%d] and msg\n", index, j);
		msgstruct *tempmsg=(msgstruct *)(PeList[index][j]->msg);
		FreeList->Add(tempmsg->msg);
		CmiFree(PeList[index][j]->msg);
	  	CmiFree(PeList[index][j]);
	}
	PeList[index][j]=NULL;
      }
      msgnum[index]=0;
  //ComlibPrintf("%d Done Packing pesize=%d\n", CmiMyPe(), pelistsize);
  }
  offset=-1;
  pehead[peheadindex++]=offset; 
  *msgvect=mvector;
  *lengths=sizes;
  return(mvectorindex);
} 
*********************************/

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


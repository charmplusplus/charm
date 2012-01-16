#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "charm++.h"

#include "receiver.h"


receiver::receiver() 
{
  msgTbl = CmmNew();
  reqTbl = CmmNew();

  callback = NULL;
  counter = -1;
  startwaiting = 0;
}

receiver::receiver(CkMigrateMessage *m)
{
  msgTbl = CmmNew();
  reqTbl = CmmNew();

  callback = NULL;
}

void receiver::pup(PUP::er &p)
{
  p(counter);
  p(startwaiting);
  // pack CmmTable: msgTbl, reqTbl
  pupCmmTable(msgTbl, p);
  pupCmmTable(reqTbl, p);
}

receiver::~receiver()
{
  CmmFree(msgTbl);
  CmmFree(reqTbl);
}

#define MYMIN(a,b) (a)<(b)?(a):(b)

// other receiver send message here (active send)
void receiver::sendTo(receiverMsg *msg, int tag, char *pointer, int size, int from, int refno)
{
  int tags[3], ret_tags[3];

//CkPrintf("Sendto (msgTbl): tag:%d, from:%d, refno:%d size:%d. \n", tag, from, refno,size);
  tags[0] = tag; tags[1] = from; tags[2] = refno;
  tblEntry *req = (tblEntry *)CmmGet(reqTbl, 3, tags, ret_tags);

  if (req) {
    //  irecv called before; copy buffer
    memcpy(req->buf, pointer, MYMIN(size, req->size)); 
    delete msg;
    delete req;

    recvAlready();
  }
  else {
    // msg come before irecv called
    tags[0] = tag; tags[1] = from; tags[2] = refno;
    req = new tblEntry;
    req->msg = msg;
    req->size = size;
    CmmPut(msgTbl, 3, tags, req);
  }

}

void receiver::generic(receiverMsg *msg)
{
  sendTo(msg, msg->tag, msg->buf, msg->size, msg->sendFrom, msg->refno);
}

void receiver::syncSend(receiverMsg *msg)
{
  sendTo(msg, msg->tag, msg->buf, msg->size, msg->sendFrom, msg->refno);
}

extern "C" int typesize(int type, int count)
{
  switch(type) {
    case CMPI_DOUBLE_PRECISION : return count*sizeof(double);
    case CMPI_INTEGER : return count*sizeof(int);
    case CMPI_REAL : return count*sizeof(float);
    case CMPI_COMPLEX: return 2*count*sizeof(double);
    case CMPI_LOGICAL: return 2*count*sizeof(int);
    case CMPI_CHARACTER:
    case CMPI_BYTE:
    case CMPI_PACKED:
    default:
      return count;
  }
}

void receiver::isend(void *buf, int count, int datatype, int dest, int tag, int refno)
{
 int size = typesize(datatype, count);
 receiverMsg * d = new (size, 0) receiverMsg;
 d->tag = tag;
 d->sendFrom = thisIndex;
 d->refno = refno;
 d->size = size;
 memcpy(d->buf, buf, size);
 CProxy_receiver B(thisArrayID);
 B[dest].generic(d);
}

void receiver::irecv(void *buf, int count, int datatype, int source, int tag, int refno)
{
  int tags[3], ret_tags[3];
  int size = typesize(datatype, count);

  tags[0] = tag; tags[1] = source; tags[2] = refno;
  tblEntry *req = (tblEntry *)CmmGet(msgTbl, 3, tags, ret_tags);

  if (req) {
    // send called before; copy buffer into
    memcpy(buf, req->msg->buf, MYMIN(size, req->size));
    delete req->msg;
    delete req;
  }
  else {
//CkPrintf("irecv (reqtbl): tag:%d, senderTag:%d, refno:%d. \n", tag, senderTag, refno);
   // recv called before send
    tags[0] = tag; tags[1] = source; tags[2] = refno;
    req = new tblEntry;
    req->buf = (char *)buf;
    req->size = size;
    CmmPut(reqTbl, 3, tags, req);
  }
}

int receiver::iAlltoAll(void *sendbuf, int sendcount, int sendtype, 
	      void *recvbuf, int recvcount, int recvtype, int refno)
{
  int nPe = ckGetArraySize();  // should be number of elements in array1D
  int tag = 65535;	// special tag
  int i;
  for (i=0; i<nPe; i++) 
      isend(((char *)sendbuf)+i*typesize(sendtype, sendcount), sendcount, sendtype, i, tag, refno);
  for (i=0; i<nPe; i++) 
      irecv(((char *)recvbuf)+i*typesize(recvtype, recvcount), recvcount, recvtype, i, tag, refno);
  return 0;
}

int receiver::iAlltoAllv(void *sendbuf, int *sendcount, int *sdispls, int sendtype, void *recvbuf, int *recvcount, int *rdispls, int recvtype, int refno)
{
  int nPe = ckGetArraySize();  // should be number of elements in array1D
  int tag = 65535;	// special tag
  int i;
  for (i=0; i<nPe; i++) 
      isend(((char *)sendbuf)+sdispls[i]*typesize(sendtype, 1), sendcount[i], sendtype, i, tag, refno);
  for (i=0; i<nPe; i++) 
      irecv(((char *)recvbuf)+rdispls[i]*typesize(recvtype, 1), recvcount[i], recvtype, i, tag, refno);
  return 0;
}

void receiver::iwaitAll(recvCallBack f, void *data, int ref)
{
  if (callback != NULL) 
  {
    CkPrintf("iwaitAll wrong!\n");
    CkExit();
  }
  callback = f;
  this->cb_data = data;
  counter = ref;

  startwaiting = 1;

  recvAlready();
}

void receiver::iwaitAll(int ref)
{
  callback = NULL;
  this->cb_data = NULL;
  counter = ref;
  startwaiting = 1;

  recvAlready();
}

void receiver::recvAlready()
{
//   if (callback == NULL) return;
   if (!startwaiting) return;

   int tags[3], ret_tags[3];
   tags[0] = CmmWildCard; tags[1] = CmmWildCard; tags[2] = counter;
   tblEntry *req1 = (tblEntry *)CmmProbe(reqTbl, 3, tags, ret_tags);
   tblEntry *req2 = (tblEntry *)CmmProbe(msgTbl, 3, tags, ret_tags);
   if (req1 == NULL && req2 == NULL && startwaiting)  //  && callback != NULL) 
   {
      startwaiting = 0;
      CProxy_receiver B(thisArrayID);
      B[thisIndex].ready2go();
//CkPrintf("[%d] ready to go on %d\n", thisIndex, counter);
   }
}

void receiver::ready2go()
{
/*
    if (callback == NULL) CkPrintf("Fatal error in Call back on [%d]. \n", thisIndex);

    recvCallBack tmpfn = callback;
    callback = NULL;
    tmpfn(cb_data);
*/
    if (callback) {
        recvCallBack tmpfn = callback;
        callback = NULL;
        tmpfn(cb_data);
    }
    else {
//CkPrintf("[%d] resumeFromWait\n", CkMyPe());
	resumeFromWait();
    }
}

void receiver::resumeFromWait()
{
    CkPrintf("Please write your own resumeFromWait\n"); 
}

void receiver::pupCmmTable(CmmTable &t, PUP::er &p)
{
    tblEntry *msg;
    int tags[3], rtags[3];
    int num = CmmEntries(t);
    p(num);
    if (p.isPacking()) {
	tags[0] = tags[1] = tags[2] = CmmWildCard;
	tblEntry *msg = (tblEntry *)CmmProbe(t, 3, tags, rtags);
	while (msg) {
	    p(rtags, 3);
	    p(msg->size);
	    p(msg->buf, msg->size);
	    msg = (tblEntry *)CmmProbe(t, 3, tags, rtags);
	}
    }
    else {
	int s;
//	t = CmmNew();
	for (int i=0; i<num; i++)  {
	    p(rtags, 3);
	    msg = new tblEntry;
	    p(msg->size);
	    msg->buf = new char[msg->size];
	    p(msg->buf, msg->size);
	}
    }
}

#include "receiver.def.h"



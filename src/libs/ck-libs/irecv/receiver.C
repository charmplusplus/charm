#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "charm++.h"

#include "receiver.h"


receiver::receiver(void) 
{
  msgTbl = CmmNew();
  reqTbl = CmmNew();

  callback = 0;
  counter = -1;
}

receiver::receiver(CkMigrateMessage *m)
{
  msgTbl = CmmNew();
  reqTbl = CmmNew();
}

void receiver::pup(PUP::er &p)
{
  ArrayElement1D::pup(p);
  p(counter);
  // pack CmmTable: msgTbl, reqTbl
  pupCmmTable(msgTbl, p);
  pupCmmTable(reqTbl, p);
}

receiver::~receiver()
{
  CmmFree(msgTbl);
  CmmFree(reqTbl);
}

#define MIN(a,b) ((a)<(b)?(a):(b))

// other receiver send message here (active send)
void receiver::sendTo(receiverMsg *m)
{
  int tags[4], ret_tags[4];
  int size = m->tags[3];
  m->tags[3] = CmmWildCard;

  void *req = CmmGet(reqTbl, 4, m->tags, ret_tags);

  if (req) {
    //  irecv called before; copy buffer
    memcpy(req, m->buf, MIN(size, ret_tags[3])); 
    delete m;
    if (callback && m->tags[2]==counter) {
      tags[0] = tags[1] = tags[3] = CmmWildCard; tags[2] = counter;
      void *req1 = CmmProbe(reqTbl, 4, tags, ret_tags);
      if (!req1) {
        recvCallBack tmpfn = callback;
        callback = 0;
        tmpfn(cb_data);
      }
    }
  } else {
    // msg came before irecv called
    m->tags[3] = size;
    CmmPut(msgTbl, 4, m->tags, m);
  }
}

extern "C" int typesize(int type, int count)
{
  switch(type) {
    case CMPI_DOUBLE_PRECISION : return count*sizeof(double);
    case CMPI_INTEGER : return count*sizeof(int);
    case CMPI_REAL : return count*sizeof(float);
    case CMPI_COMPLEX: return 2*count*sizeof(double);
    case CMPI_LOGICAL: return 2*count*sizeof(int);
    case CMPI_CHAR: return count;
    case CMPI_BYTE: return count;
    case CMPI_PACKED: return count;
    default:
      CkAbort("Type not supported\n");
      return 0;
  }
}

void 
receiver::isend(void *buf, int count, int datatype, int dest, int tag, 
                int refno)
{
  //CkPrintf("[%d] isend to %d with ref %d\n", thisIndex, dest, refno);
  int size = typesize(datatype, count);
  receiverMsg *d = new (&size, 0) receiverMsg;
  d->tags[0] = tag;
  d->tags[1] = thisIndex;
  d->tags[2] = refno;
  d->tags[3] = size;
  memcpy(d->buf, buf, size);
  CProxy_receiver B(thisArrayID);
  B[dest].sendTo(d);
}

void 
receiver::irecv(void *buf, int count, int datatype, int source, int tag, 
                int refno)
{
  //CkPrintf("[%d] irecv from %d with ref %d\n", thisIndex, source, refno);
  int tags[4], ret_tags[4];
  int size = typesize(datatype, count);

  tags[0] = tag; tags[1] = source; tags[2] = refno; tags[3] = CmmWildCard;
  receiverMsg *msg = (receiverMsg *)CmmGet(msgTbl, 4, tags, ret_tags);

  if (msg) {
    // send called before; copy buffer into
    memcpy(buf, msg->buf, MIN(size, msg->tags[3]));
    delete msg;
  } else {
   // recv called before send
    tags[3] = size;
    CmmPut(reqTbl, 4, tags, buf);
  }
}

#define GATHER_TAG   65535;

int 
receiver::iAlltoAll(void *sendbuf, int sendcount, int sendtype, 
	            void *recvbuf, int recvcount, int recvtype, int refno)
{
  int nPe = getArraySize();  // should be number of elements in array1D
  int tag = GATHER_TAG;	// special tag
  int i;
  for (i=0; i<nPe; i++) {
      isend(((char *)sendbuf)+i*typesize(sendtype, sendcount), 
            sendcount, sendtype, i, tag, refno);
  }
  for (i=0; i<nPe; i++)  {
      irecv(((char *)recvbuf)+i*typesize(recvtype, recvcount), 
            recvcount, recvtype, i, tag, refno);
  }
  return 0;
}

int 
receiver::iAlltoAllv(void *sendbuf, int *sendcount, int *sdispls, int sendtype,
                     void *recvbuf, int *recvcount, int *rdispls, int recvtype,
                     int refno)
{
  int nPe = getArraySize();  // should be number of elements in array1D
  int tag = GATHER_TAG;	// special tag
  int i;
  for (i=0; i<nPe; i++) {
      isend(((char *)sendbuf)+sdispls[i]*typesize(sendtype, 1), 
            sendcount[i], sendtype, i, tag, refno);
  }
  for (i=0; i<nPe; i++)  {
      irecv(((char *)recvbuf)+rdispls[i]*typesize(recvtype, 1), 
            recvcount[i], recvtype, i, tag, refno);
  }
  return 0;
}

void receiver::iwaitAll(recvCallBack f, void *data, int ref)
{
  //CkPrintf("[%d] iwaitall with ref %d\n", thisIndex, ref);
  int tags[4], ret_tags[4];
  tags[0] = tags[1] = tags[3] = CmmWildCard; tags[2] = ref;
  void *req1 = CmmProbe(reqTbl, 4, tags, ret_tags);
  if (req1) { // some unsatisfied irecv requests exist
    callback = f;
    cb_data = data;
    counter = ref;
  } else { // all irecv requests for refnum ref satisfied
    f(data);
  }
}

void receiver::pupCmmTable(CmmTable &t, PUP::er &p)
{
  void *msg;
  int tags[4], rtags[4];
  int num = CmmEntries(t);
  p(num);
  if (p.isPacking()) {
    tags[0] = tags[1] = tags[2] = tags[3] = CmmWildCard;
    while (msg = CmmProbe(t, 4, tags, rtags)) {
      p(rtags, 4);
      p(msg, rtags[3]);
    }
  } else {
    for (int i=0; i<num; i++)  {
      p(tags, 4);
      msg = new char[tags[3]];
      p(msg, tags[3]);
      CmmPut(t, 4, tags, msg);
    }
  }
}

extern "C" double IMPI_Wtime(void)
{
  return CmiWallTimer();
}

extern "C" double impi_wtime_(void)
{
  return IMPI_Wtime();
}

#include "receiver.def.h"




#ifndef _RECEIVER_H
#define _RECEIVER_H

#include "converse.h"
#include "receiver.decl.h"

typedef void (* recvCallBack)(void *);

#define CMPI_DOUBLE_PRECISION 0
#define CMPI_INTEGER 1
#define CMPI_REAL 2
#define CMPI_COMPLEX 3
#define CMPI_LOGICAL 4
#define CMPI_CHAR 5
#define CMPI_BYTE 6
#define CMPI_PACKED 7

class receiverMsg: public CMessage_receiverMsg
{
public:
  int tags[4]; // tag; sendFrom; refno; size;
  char *buf;

  static void *alloc(int mnum, size_t size, int *sizes, int pbits)
  {
    receiverMsg *m = (receiverMsg *) CkAllocMsg(mnum, size+sizes[0], pbits);
    m->buf = (char *)((char *)m + size);
    return (void *)m;
  }

  static void *pack(receiverMsg *m)
  {
    m->buf = (char *)((char *)m->buf - (char *)&m->buf);
    return (void *)m;
  }

  static receiverMsg *unpack(void *buf)
  {
    receiverMsg *m = (receiverMsg *) buf;
    m->buf = (char *)((char*)&(m->buf) + (size_t)(m->buf));
    return m;
  }
};


class receiver: public ArrayElement1D
{
private:
  CmmTable  msgTbl;
  CmmTable  reqTbl;

  int counter;
  recvCallBack callback;
  void *cb_data;

public:
  receiver(void);
  receiver(CkMigrateMessage *);
  ~receiver();

  // interface
  void isend(void *buf, int count, int datatype, int dest, int tag, int refno);
  void irecv(void *buf, int count, int datatype, int source, int tag, int refno);   
  int iAlltoAll(void *sendbuf, int sendcount, int sendtype, 
		 void *recvbuf, int recvcount, int recvtype, int refno);
  int iAlltoAllv(void *sendbuf, int *sendcount, int *sdispls, int sendtype, 
	 void *recvbuf, int *recvcount, int *rdispls, int recvtype, int refno);
  void iwaitAll(int refno);
  void iwaitAll(recvCallBack f, void *data, int refno);     // for fortran

  //entry 
  void sendTo(receiverMsg *m);

private:
  void pupCmmTable(CmmTable &t, PUP::er &p);

protected:
  void pup(PUP::er &p);
};

#endif

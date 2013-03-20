
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
#define CMPI_CHARACTER 5
#define CMPI_BYTE 6
#define CMPI_PACKED 7

class receiverMsg: public CMessage_receiverMsg
{
public:
  int tag;
  int sendFrom;
  int refno;
  int size;
  char *buf;

/*
  static void *alloc(int mnum, size_t size, int *sizes, int pbits)
  {
    int stmp = sizes[0]*sizeof(char);
    receiverMsg *m = (receiverMsg *) CkAllocMsg(mnum, size+stmp, pbits);
    m->size = sizes[0];
    m->buf = (char *)((char *)m + size);
    return (void *)m;
  }

  static void *pack(receiverMsg *m)
  {
//    return (void *)m;
    m->buf = (char *)((char *)m->buf - (char *)&m->buf);
    return (void *)m;
  }

  static receiverMsg *unpack(void *buf)
  {
    receiverMsg *m = (receiverMsg *) buf;
//    m->buf = (char *)((char *)m+sizeof(receiverMsg));
    m->buf = (char *)((char*)&(m->buf) + (size_t)(m->buf));
    return m;
  }
*/
};


class receiver: public CBase_receiver
{
private:
  CmmTable  msgTbl;
  CmmTable  reqTbl;
  int counter;
  int startwaiting;

  recvCallBack callback;
  void *cb_data;

  typedef struct _tblEntry {
    receiverMsg *msg;
    char *buf;
    int size;
  } tblEntry;


public:
  receiver();
  receiver(CkMigrateMessage *);
  ~receiver();
  void pup(PUP::er &p);

  // interface
  void isend(void *buf, int count, int datatype, int dest, int tag, int refno);
  void irecv(void *buf, int count, int datatype, int source, int tag, int refno);   
  int iAlltoAll(void *sendbuf, int sendcount, int sendtype, 
		 void *recvbuf, int recvcount, int recvtype, int refno);
  int iAlltoAllv(void *sendbuf, int *sendcount, int *sdispls, int sendtype, 
	 void *recvbuf, int *recvcount, int *rdispls, int recvtype, int refno);
  void iwaitAll(int refno);
  void iwaitAll(recvCallBack f, void *data, int refno);     // for fortran

  // entry
  void generic(receiverMsg *);
  void syncSend(receiverMsg *);
  void ready2go();

private:
  void sendTo(receiverMsg *, int tag, char *pointer, int size, int from, int refno);
  void recvAlready();
  void pupCmmTable(CmmTable &t, PUP::er &p);

protected:
  virtual void resumeFromWait();
};

#endif

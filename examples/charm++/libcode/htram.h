#ifndef __HTRAM_H__
#define __HTRAM_H__
#include "htram.decl.h"
/* readonly */ extern CProxy_HTram htramProxy;
/* readonly */ extern CProxy_HTramRecv nodeGrpProxy;

using namespace std;
#define BUFSIZE 1024

typedef struct item {
  int destPe;
  int payload;
} itemT; //make customized size

class HTramMessage : public CMessage_HTramMessage {
  public:
    HTramMessage() {next = 0;}
    HTramMessage(int size, itemT *buf): next(size) {
      std::copy(buf, buf+size, buffer);
    }
    itemT buffer[BUFSIZE];
    int next; //next available slot in buffer
};

typedef void (*callback_function)(int);

class HTram : public CBase_HTram {
  HTram_SDAG_CODE

  private:
    callback_function cb;
    CkCallback endCb;
    int myPE;
    HTramMessage **msgBuffers;
  public:
    HTram(CkCallback cb);
    HTram(CkMigrateMessage* msg);
    void setCb(void (*func)(int));
    int getAggregatingPE(int dest_pe);
    void insertValue(int send_value, int dest_pe);
    void tflush();
    void receivePerPE(HTramMessage *);
};


class HTramRecv : public CBase_HTramRecv {
  HTramRecv_SDAG_CODE

  public:
    HTramRecv();
    HTramRecv(CkMigrateMessage* msg);
    void receive(HTramMessage*);
};
#endif

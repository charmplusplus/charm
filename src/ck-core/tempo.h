#ifndef TEMPO_H
#define TEMPO_H

#include "charm++.h"
#include <stdlib.h>
#include "tempo.decl.h"

class TempoMessage : public ArrayMessage, public CMessage_TempoMessage
{
  public:
    int tag, length;
    void *data;
  
    TempoMessage(void) { data = 0; length = 0; }
    TempoMessage(int t, int l, void *d) : tag(t), length(l) {
      data = malloc(l);
      memcpy(data, d, l);
    }
    ~TempoMessage() {
      if(data) free(data);
    }
    static void *pack(TempoMessage *);
    static TempoMessage *unpack(void *);
};

class Tempo 
{
  CmmTable tempoMessages;
  int sleeping;
  CthThread thread_id;
  
  public :
    Tempo();
    void ckTempoRecv(int tag, void *buffer, int buflen);
    static void ckTempoSend(CkChareID chareid,int tag,void *buffer,int buflen);
    void tempoGeneric(TempoMessage *themsg);
    int ckTempoProbe(int tag);
};

class TempoChare : public Chare, public Tempo
{
  public:
    TempoChare(void) {};
};

class TempoGroup : public Group, public Tempo
{
  public :
    TempoGroup(void) {};
    static void ckTempoBcast(int sender, int bocid, int tag, 
                             void *buffer, int buflen);
    static void ckTempoSendBranch(int bocid, int tag, void *buffer, int buflen,
                                  int processor);
    void ckTempoBcast(int sender, int tag, void *buffer, int buflen);
};

class TempoArray : public ArrayElement, public Tempo
{
  public:
    TempoArray(ArrayElementCreateMessage *msg) : ArrayElement(msg)
      { finishConstruction(); }
    TempoArray(ArrayElementMigrateMessage *msg) : ArrayElement(msg)
      { finishMigration(); }
    static void ckTempoSendElem(CkAID aid, int tag, void *buffer, int buflen,
                                  int idx);
    void ckTempoSendElem(int tag, void *buffer, int buflen, int idx);
};

#endif

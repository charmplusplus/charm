#ifndef COMMLIBMANAGER_H
#define COMMLIBMANAGER_H

#include "charm++.h"
#include "converse.h"
#include "envelope.h"
#include "commlib.h"

#include "ComlibModule.decl.h"

#define USE_DIRECT 0
#define USE_TREE 1
#define USE_MESH 2
#define USE_HYPERCUBE 3
#define USE_GRID 4
#define NAMD_STRAT 5

#define GROUP_SEND 0
#define ARRAY_SEND 1
#define CHARE_SEND 2
#define RECV 4

class CharmMessageHolder {
 public:
    int dest_proc;
    char *data;
    CharmMessageHolder *next; // also used for the refield at the receiver

    CharmMessageHolder(char * msg, int dest_proc);
    char * getCharmMessage();
    void copy(char *buf);
    int getSize();
    void init(char *root);
    void setRefcount(char * root_msg);
};

class DummyMsg: public CMessage_DummyMsg {
    int dummy;
};

class ComlibMsg: public CMessage_ComlibMsg {
 public:
    int nmessages;
    int curSize;
    int destPE;
    char* data;
    
    void insert(CharmMessageHolder *msg);
    CharmMessageHolder * next();
};

class ComlibManager: public CkDelegateMgr{
    
    CharmMessageHolder * messageBuf;

    int *procMap;

    int createDone, doneReceived;

    int npes;
    int *pelist;

    int messagesBeforeFlush;
    int bytesBeforeFlush;

    int messageCount;

    int nelements; //number of array elements on one processor
    int elementCount; //counter for the above
    int strategy;
    comID comid;
    
    //flags
    int idSet, iterationFinished;
    
    void init(int s, int n, int Messages, int nBytes); //strategy, nelements 
    void setReverseMap(int *, int);

 public:
    ComlibManager(int s); //strategy, nelements 
    ComlibManager(int s, int n); //strategy, nelements 
    ComlibManager(int s, int nMessages, int nBytes); //strategy, nelements 
    ComlibManager(int s, int n, int nMessages, int nBytes); //strategy, nelements 

    void done();
    void localElement();

    void receiveID(comID id);
    void receiveID(int npes, int *pelist, comID id);

    void ArraySend(int ep, void *msg, const CkArrayIndexMax &idx, CkArrayID a);
    void GroupSend(int ep, void *msg, int onpe, CkGroupID gid);
    void setNumMessages(int nmessages);
    
    void beginIteration();
    void endIteration();

    void receiveNamdMessage(ComlibMsg * msg);
    void createId();
    void createId(int *, int);
};

#endif

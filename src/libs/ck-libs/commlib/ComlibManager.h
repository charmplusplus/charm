#include "charm++.h"
#include "converse.h"
#include "envelope.h"
#include "commlib.h"

#include "ComlibModule.decl.h"

class CharmMessageHolder {
 public:
    int ep;
    CkArrayIndexMax idx;
    CkArrayID a;
    int size;
    char *data;
    CharmMessageHolder *next;

    CharmMessageHolder(int ep, void *charm_message, 
                       CkArrayIndexMax &idx, CkArrayID a);
    void copy(char *buf);
    void init();
    int getSize();
    CkArrayMessage * getCharmMessage();
    void CharmMessageHolder::setRefcount(void *msg);
};

class ComlibMsg: public CMessage_ComlibMsg {
 public:
    int nmessages;
    int curSize;
    int src;
    int isDummy;
    char *data;

    static void *alloc(int mnum, size_t size, int *sizes, int priobits);
    static void *pack(ComlibMsg *msg);
    static ComlibMsg *unpack(void *buf);
    
    void insert(CharmMessageHolder *msg);
    CharmMessageHolder * next();
};

class ComlibManager: public CkDelegateMgr{
    
    CharmMessageHolder ** messageBuf;
    ComlibMsg ** receiveBuf;

    int *messageCount;
    int *messageSize;
    //    int nMessages;
    int nelements; //number of array elements on one processor
    int elementCount; //counter for the above
    int strategy;
    comID comid;
    
    //flags
    int idSet, iterationFinished;
    
    void sendMessage(int dest_proc);
    
 public:
    ComlibManager(int s); //strategy, nelements 
    //    ComlibManager(int s, int n); //strategy, nelements 

    void done();
    void localElement();
    void receiveID(comID id);
    void ArraySend(int ep, void *msg, const CkArrayIndexMax &idx, CkArrayID a);
    void receiveMessage(ComlibMsg *);
    void setNumMessages(int nmessages);

    void beginIteration();
    void endIteration();
};

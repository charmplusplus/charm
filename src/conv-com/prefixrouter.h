#ifndef PREFIX_ROUTER_H
#define PREFIX_ROUTER_H

#include <math.h>
#include <converse.h>
#include "convcomlib.h"

//Prefix router to avoid contention on m,n-tree networks

class PrefixRouter : public Router {
    int *gpes;
    int *prefix_pelist;
    int npes, MyPe;

 public:
    PrefixRouter(int _npes, int me) : npes(_npes), MyPe(me) {};
    virtual ~PrefixRouter() {};
    
    virtual void EachToManyMulticastQ(comID id, CkQ<MessageHolder *> &msgq);

    //communication operation
    virtual void SetMap(int *pelist) {gpes = pelist;}
    virtual void sendMulticast(CkQ<MessageHolder *> &msgq);
    virtual void sendPointToPoint(CkQ<MessageHolder *> &msgq);
};

#endif

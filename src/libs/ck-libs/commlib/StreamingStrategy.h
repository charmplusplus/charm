#ifndef STREAMING_STRATEGY
#define STREAMING_STRATEGY
#include "ComlibManager.h"

class StreamingStrategy : public Strategy {
    CharmMessageHolder **streamingMsgBuf;
    int *streamingMsgCount;
    int PERIOD;

 public:
    StreamingStrategy(int period);
    StreamingStrategy(CkMigrateMessage *){}
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();
    void periodicFlush();

    virtual void pup(PUP::er &p);
    PUPable_decl(StreamingStrategy);
};
#endif

#include "ComlibManager.h"

class StreamingStrategy : public Strategy {
    CharmMessageHolder **streamingMsgBuf;
    int *streamingMsgCount;
    int PERIOD;

 public:
    StreamingStrategy(int period);
    void insertMessage(CharmMessageHolder *msg);
    void doneInserting();
};

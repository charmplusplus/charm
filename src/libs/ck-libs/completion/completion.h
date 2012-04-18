#ifndef COMPLETION_H
#define COMPLETION_H

#include "completion.decl.h"

class CompletionDetector : public CBase_CompletionDetector {
public:
    CompletionDetector();

    // Local methods
    void produce(int events_produced = 1);
    void consume(int events_consumed = 1);
    void done(int producers_done = 1);

    CompletionDetector_SDAG_CODE

private:
    int produced, consumed, unconsumed;
    int producers_total, producers_done_local, producers_done_global;
    int prio;
    bool running;

    void init();
};

#endif

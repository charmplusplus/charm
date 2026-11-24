#ifndef __COMPUTE_H__
#define __COMPUTE_H__

#include "defs.h"

//class representing the interaction agents between a couple of cells
class Compute : public CBase_Compute {
  private:
    Compute_CORO_CODE
    int stepCount;  //current step number
    ParticleDataMsg *bufferedMsg; //first message received for interaction, held across the second wait
    ParticleDataMsg *msg;         //second (or self) message, consumed right after its wait

  public:
    Compute();
    Compute(CkMigrateMessage *msg);
    void pup(PUP::er &p);

    void selfInteract(ParticleDataMsg *msg);
    void interact(ParticleDataMsg *msg);
    ck::coroutine run();
};

#endif

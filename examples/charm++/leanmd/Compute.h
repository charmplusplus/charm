#ifndef __COMPUTE_H__
#define __COMPUTE_H__

#include "defs.h"

//class representing the interaction agents between a couple of cells
class Compute : public CBase_Compute {
  private:
    Compute_SDAG_CODE
    int stepCount;  //current step number
    ParticleDataMsg *bufferedMsg; //copy of first message received for interaction

  public:
    Compute();
    Compute(CkMigrateMessage *msg);
    void pup(PUP::er &p);

    void selfInteract(ParticleDataMsg *msg);
    void interact(ParticleDataMsg *msg);
};

#endif

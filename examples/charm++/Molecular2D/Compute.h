/** \file Compute.h
 *  Author: Abhinav S Bhatele
 *  Date Created: July 1st, 2008
 *
 */

#ifndef __COMPUTE_H__
#define __COMPUTE_H__

#include "common.h"

// Class representing the interaction agents between a couple of cells
class Compute : public CBase_Compute {
  private:
    int cellCount;  // to count the number of interact() calls
    std::vector<Particle> bufferedParticles;
    int bufferedX;
    int bufferedY;

    void interact(std::vector<Particle> &first, std::vector<Particle> &second);
    void interact(Particle &first, Particle &second);

  public:
    Compute();
    Compute(CkMigrateMessage *msg);

    void interact(std::vector<Particle> particles, int i, int j);

};

#endif

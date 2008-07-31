/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/** \file Patch.h
 *  Author: Abhinav S Bhatele
 *  Date Created: July 1st, 2008
 *
 */


#ifndef __PATCH_H__
#define __PATCH_H__

/** \class Main
 *
 */
class Main : public CBase_Main {
  private:
    int checkInCount; // Count to terminate

  public:
    Main(CkArgMsg* msg);
    Main(CkMigrateMessage* msg);

    void checkIn();
};

/** \class Patch
 *  Class representing a cell in the grid. 
 *  We consider each cell as a square of LxL units
 */
class Patch : public CBase_Patch {
  private:
    CkVec<Particle> particles;
    CkVec<Particle> incomingParticles;
    int forceCount;		// to count the returns from interactions
    int stepCount;		// to count the number of steps, and decide when to stop
    int updateCount;
    bool updateFlag;
    bool incomingFlag;

    void updateProperties();	// updates properties after receiving forces from computes
    void checkNextStep();	// checks whether to continue with next step
    void print();		// prints all its particles

  public:
    Patch();
    Patch(CkMigrateMessage *msg);
    ~Patch();

    void start();
    void updateParticles(CkVec<Particle>&);
    void updateForces(CkVec<Particle>&);
    void limitVelocity(Particle&);
    Particle& wrapAround(Particle &);
    // void requestNextFrame(liveVizRequestMsg *m);
};

#endif

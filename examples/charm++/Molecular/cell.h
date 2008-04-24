/*
 University of Illinois at Urbana-Champaign
 Department of Computer Science
 Parallel Programming Lab
 2008
*/

#ifndef __CELL_H__
#define __CELL_H__

#include "common.h"

// Class representing a cell in the grid. We consider each cell as a square of LxL units
class Cell : public CBase_Cell {
  private:
    CkVec<Particle> particles;
    CkVec<Particle> incomingParticles;
    int forceCount; // to count the returns from interactions
    int stepCount;  // to count the number of steps, and decide when to stop
		int updateCount;
		bool updateFlag;
		bool incomingFlag;

		void updateProperties();  // updates properties after receiving forces from interactions
		void checkNextStep();		  // checks whether to continue with next step
		void print();						  // prints all its particles

  public:
    Cell();
    Cell(CkMigrateMessage *msg);
    ~Cell();

    void start();
    void updateParticles(CkVec<Particle>&);
    void updateForces(CkVec<Particle>&);
    void limitVelocity(Particle&);
    Particle& wrapAround(Particle &);
    void stepDone();
    void requestNextFrame(liveVizRequestMsg *m);
};

// Class representing the interaction agents between a couple of cells
class Interaction : public CBase_Interaction {
  private:
    int cellCount;  // to count the number of interact() calls
    CkVec<Particle> bufferedParticles;
    int bufferedX;
 		int bufferedY;

		void interact(CkVec<Particle> &first, CkVec<Particle> &second);
		void interact(Particle &first, Particle &second);

  public:
    Interaction();
    Interaction(CkMigrateMessage *msg);

    void interact(CkVec<Particle> particles, int i, int j);


};

#endif

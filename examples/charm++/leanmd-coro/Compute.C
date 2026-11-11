#include "defs.h"
#include "leanmd.decl.h"
#include "ckcoro.h"
#include "Cell.h"
#include "Compute.h"
#include "physics.h"
#include <algorithm>
using std::swap;

extern /* readonly */ CProxy_Main mainProxy;
extern /* readonly */ CkGroupID mCastGrpID;

extern /* readonly */ int cellArrayDimX;
extern /* readonly */ int cellArrayDimY;
extern /* readonly */ int cellArrayDimZ;
extern /* readonly */ int finalStepCount; 

//compute - Default constructor
Compute::Compute() {
  stepCount = 1;
  bufferedMsg = msg = nullptr;
  usesAtSync = true;
}

Compute::Compute(CkMigrateMessage *m): CBase_Compute(m)  {
  usesAtSync = true;
  bufferedMsg = msg = nullptr;
  delete m;
}

// The SDAG run() of the original. The coroutine owns the messages it
// receives: selfInteract and interact (calcInternalForces and
// calcPairForces in physics.h) delete them.
ck::coroutine Compute::run() {
  CK_COROUTINE();
  for(stepCount = 1; stepCount <= finalStepCount; stepCount++) {
    //self interaction check
    if(thisIndex.x1==thisIndex.x2 && thisIndex.y1==thisIndex.y2 && thisIndex.z1==thisIndex.z2) {
      CK_WHEN(inbox::calculateForces(stepCount), msg);   // when calculateForces[stepCount](ParticleDataMsg *msg)
      selfInteract(msg);
      msg = nullptr;
    } else {
      //receive positions from two cells - buffer the first one
      CK_WHEN(inbox::calculateForces(stepCount), bufferedMsg);
      CK_WHEN(inbox::calculateForces(stepCount), msg);
      interact(msg);
      bufferedMsg = msg = nullptr;
    }
  }
  CK_END();
}

//interaction within a cell
void Compute::selfInteract(ParticleDataMsg *msg){
  calcInternalForces(msg, stepCount);
}

//interaction between two cells
void Compute::interact(ParticleDataMsg *msg){
  calcPairForces(msg, bufferedMsg, stepCount);
}

//pack important information if I am moving
void Compute::pup(PUP::er &p) {
  p | stepCount;
  // The first message of a pair is held across the wait for the second.
  bool hasBuffered = (bufferedMsg != nullptr);
  p | hasBuffered;
  if (hasBuffered) CkPupMessage(p, (void **)&bufferedMsg);
  else if (p.isUnpacking()) bufferedMsg = nullptr;
  if (p.isUnpacking()) msg = nullptr;
}

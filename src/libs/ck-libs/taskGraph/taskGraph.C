/*
 * taskGraph: ordered execution of repetitive computation
 */

#include "taskGraph.h"
#include "taskGraph.decl.h"


CkArrayID taskGraphInit() {
  return CProxy_taskGraphArray::ckNew();
}

void taskGraphDelete(CkArrayID id, CkArrayIndex taskID) {
  CProxy_taskGraphArray array(id);
  array(taskID).deleteElement();
}

/*
 * Now define the taskGraphArray that actually handles doing all that work.
 */
taskGraphArray::taskGraphArray(
	CkVec<CkArrayIndex> deps,
	taskGraphSolver *data,
	CkCallback returnResults
) : Waiting() {
  // Set some state variables
  ReturnResults = returnResults;
  Self = data;
  isSolved = false;

  // Save everything I need to know about
  DepsCount = deps.length();
  DepsData = new taskGraphSolver*[DepsCount];
  DepsReceived = 0;

  // Ask everyone I depend on for their data
  CProxy_taskGraphArray neighbor(thisArrayID);
  for ( int i = 0 ; i < DepsCount ; i++ ) {
    neighbor(deps[i]).requestData(thisIndexMax);
  }

  // If we're waiting on nothing we're solved
  tryToSolve();
}


void taskGraphArray::tryToSolve() {
  if ( DepsCount == DepsReceived ) {
    if ( DepsCount != 0 ) {
      Self->solve(DepsCount, DepsData);
    } else {
      Self->setup();
    }
    isSolved = true;

    // Return that to whoever spawned me
    callbackMsg *res = new callbackMsg( Self, thisIndexMax );
    ReturnResults.send( res );
  
    // And tell everyone who's waiting on me that I solved myself
    for ( int i = 0 ; i < Waiting.size() ; i++ ) {
      CProxy_taskGraphArray neighbor(thisArrayID);
      neighbor(Waiting[i]).depositData(Self);
    }
  }
}


void taskGraphArray::requestData(CkArrayIndex from) {
  // If the problem isn't solved, kick this request onto the waiting queue
  if ( ! isSolved ) {
    Waiting.insertAtEnd(from);
    return;
  }

  // Otherwise, if the problem is solved send them the data we generated
  CProxy_taskGraphArray neighbor(thisArrayID);
  neighbor(from).depositData(Self);
}


void taskGraphArray::depositData(taskGraphSolver *data) {
  // Someone sent me data back.
  DepsData[DepsReceived++] = data;

  // Now that we got that data try and solve the problem
  tryToSolve();
}


void taskGraphArray::deleteElement() {
  // Warn of a possibly stupid action
  if ( Waiting.size() != 0 ) {
    ckerr << "Warning! taskGraphArray::delete called on an element that "
          << "currently has " << Waiting.length() << " other elements "
	  << "waiting on it! Deleting anyway..." << endl;
  }

  // Impelemnt deletion
  CProxy_taskGraphArray array(thisArrayID);
//  array(thisIndexMax).destroy();
}

#include "taskGraph.def.h"

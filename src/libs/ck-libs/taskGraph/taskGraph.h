/*
 */
#ifndef __UIUC_CHARM_TASKGRAPH_H
#define __UIUC_CHARM_TASKGRAPH_H

#include <charm++.h>

class taskGraphSolver; // forward decl so taskGraph.decl.h doesn't die

#include "taskGraph.decl.h"

/*
 * taskGraphSolver: A parent solver class that users will sub-class off
 * of to write their solvers.
 *
 * Users must implement the pup and solve methods in their classes.
 */
class taskGraphSolver : public PUP::able {
private:
  CkArrayID __taskSet;
  CkArrayIndex __taskIndex;
  CkVec<CkArrayIndex> __taskDeps;
public:
  virtual void dependsOn(int x) { dependsOn(CkArrayIndex1D(x)); }
  virtual void dependsOn(int x, int y) { dependsOn(CkArrayIndex2D(x,y)); }
  virtual void dependsOn(int x, int y, int z) { dependsOn(CkArrayIndex3D(x,y,z)); }
  virtual void dependsOn(CkArrayIndex taskDep) { __taskDeps.push_back(CkArrayIndex(taskDep)); }

  static CkArrayID newTaskGraph() { return CProxy_taskGraphArray::ckNew(); }
  virtual void startTask() {
    CProxy_taskGraphArray array(__taskSet);
    array(__taskIndex).insert(__taskDeps, (taskGraphSolver *)this, CkCallback::ignore);
  }
  virtual void removeTask() {
    CProxy_taskGraphArray array(__taskSet);
    array(__taskIndex).deleteElement();
  }

public:
  taskGraphSolver(CkArrayID set, int x)
    : __taskDeps(), __taskSet(set), __taskIndex(CkArrayIndex1D(x)) {};
  taskGraphSolver(CkArrayID set, int x, int y)
    : __taskDeps(), __taskSet(set), __taskIndex(CkArrayIndex2D(x,y)) {};
  taskGraphSolver(CkArrayID set, int x, int y, int z)
    : __taskDeps(), __taskSet(set), __taskIndex(CkArrayIndex3D(x,y,z)) {};
  taskGraphSolver(CkArrayID set, CkArrayIndex taskIndex)
    : __taskDeps(), __taskSet(set), __taskIndex(taskIndex) {};
  taskGraphSolver(CkMigrateMessage *m) : PUP::able(m) {};

  virtual void pup(PUP::er &p) {
  }
  PUPable_abstract(taskGraphSolver);

  virtual void solve(int depsCount, taskGraphSolver *data[]) = 0;
  virtual void setup() = 0;
};



class callbackMsg : public CMessage_callbackMsg {
public:
  PUPable_marshall<taskGraphSolver> Data;
  CkArrayIndex Index;
  callbackMsg(taskGraphSolver* self, CkArrayIndex ind)
    : Data(self), Index(ind) {};
};



/*
 * Start the taskGraph abstraction. This function returns a new CkArrayID
 * of the array it creates to handle this abstraction. A user's code should
 * call this function then save the CkArrayID to pass to the other
 * taskGraph functions.
 */
CkArrayID taskGraphInit();

/*
 * Add a new task to the task graph. The array ID, task number to add,
 * and dependancies must all be specified at this time. A function which
 * will do the actual evaluation must also be specified.
 *
 * There is also a field for any static data that this task should be passed.
 */
template <class T>
void taskGraphAdd(CkArrayID id, T taskID,
		  CkVec<T> deps,
		  taskGraphSolver *self,
		  CkCallback returnResults = CkCallback::ignore) {
  CkVec<CkArrayIndex> newDeps;

  for ( int i = 0 ; i < deps.length() ; i++ ) {
    newDeps.push_back( CkArrayIndex(deps[i]) );
  }

  CkArrayIndex newTaskID( taskID );
  CProxy_taskGraphArray array(id);
  array(newTaskID).insert(newDeps, self, returnResults);
}

/*
 * Delete an old task. The array ID and task number to delete are required.
 */
void taskGraphDelete(CkArrayID id, CkArrayIndex taskID);


/*
 * Define the taskGraphArray that handles that work.
 */
class taskGraphArray : public ArrayElementMax {
protected:
  taskGraphSolver *Self;
  int isSolved;
  CkVec<CkArrayIndex> Waiting;

  int DepsCount;
  taskGraphSolver **DepsData;
  int DepsReceived;

  CkCallback ReturnResults;

  void tryToSolve();

public:
  taskGraphArray(CkVec<CkArrayIndex> deps,
                 taskGraphSolver *data,
		 CkCallback returnResults);
  taskGraphArray(CkMigrateMessage *m) {};
  void requestData(CkArrayIndex from);
  void depositData(taskGraphSolver *data);
  void deleteElement();
};

#endif /* def(thisHeader) */

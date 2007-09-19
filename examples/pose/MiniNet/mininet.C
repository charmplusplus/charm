#include <unistd.h>
#include <math.h>
#include "pose.h"
#include "mininet.h"
#include "MiniNet.def.h"
#include "Node_sim.h"

main::main(CkArgMsg *m)
{ 
  CkGetChareID(&mainhandle);

  POSE_init();

  // create all the nodes
  nodeMsg *nm;
  for (int i=0; i<4; i++) {
    nm = new nodeMsg;
    // this mapping gives us:
    // 0 --- 1
    // |     |
    // |     |
    // 3 --- 2
    nm->nbr1 = (i+1)%4;
    nm->nbr2 = (i+3)%4;
    nm->Timestamp(0);
    (*(CProxy_Node *) &POSE_Objects)[i].insert(nm);
  }
  POSE_Objects.doneInserting();
}

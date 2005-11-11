/* 

ParFUM Collision Interface

Author: Isaac Dooley 11-11-2005

*/

#include "fem_collide.h"



collide_t ParFUM_Collide_Init(int dimension){
  
  // Determine Grid Sizing from nodal coordinates
  
  double gridStart[3], gridSize[3];
  gridStart[0] = gridStart[1] = gridStart[2] = 0.0;
  gridSize[0] = gridSize[1] = gridSize[2] = 10.0;
  
  // Call COLLIDE_Init()
    c=COLLIDE_Init(MPI_COMM_WORLD, gridStart, gridSize);

}



int ParFUM_Collide(collide_t c){
  
  // Create Bounding boxes for elements, and priority array
  double *boxes = new double[nboxes*6];
  int *priorities = new int[nboxes];
  int ncoll;
  
  // Call COLLIDE_Boxes_prio()


  return 0; 
}



void ParFUM_Collide_GetCollisions(collide_t c, void* results){
  
  // Somehow transmit data
  // probably we'll do some series of MPI_Sends and MPI_Recv's in a valid ordering
  // this data will end up in results

  
}


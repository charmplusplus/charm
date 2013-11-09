/* 

ParFUM Collision Interface

Author: Isaac Dooley 11-11-2005

*/
#include "ParFUM.h"
#include "ParFUM_internals.h"

#define NULL 0

ParFUM_collider ParFUM_Collide_Init(int dimension){
  ParFUM_collider c;
  c.collide_grid = NULL; // This will be created here if we could reuse the grid
  c.dimension = dimension;
  return c;
}


int ParFUM_Collide(ParFUM_collider *c, double box_padding){
  // count collidable elements
  c->numCollidableElements = 0;

  double *boxes = new double[c->numCollidableElements];
  int *priorities = new int[c->numCollidableElements];
  int ncoll;

  // scan collidable elements, 
  // ---build the element to BoxMapping
  c->boxToElementMapping = new unsigned int[c->numCollidableElements];
  // ---find bounding boxes and priority for each element

  // ---keep track of min/max nodal coordinates


  // determine grid sizing based on the nodal coordinate ranges
  double gridStart[3], gridSize[3];
  gridStart[0] = gridStart[1] = gridStart[2] = 0.0;
  gridSize[0] = gridSize[1] = gridSize[2] = 10.0+box_padding;
  
  // Call COLLIDE_Init()
  c->collide_grid=COLLIDE_Init(MPI_COMM_WORLD, gridStart, gridSize);
    
  // Call COLLIDE_Boxes_prio()
  
  // clean up arrays which can now be deallocated
  delete [] boxes;
  delete [] priorities;
  // return number of collisions
  return 0; 
}



void ParFUM_Collide_GetCollisions(ParFUM_collider *c, void* results){


  //  Build list of collisions sorted by remote processor

  //  Copy local data into results

  //  iterate through processors, and package data for any processor with which a collision occurs

  // If lowest unprocessed one is to a greater VP, send it, otherwise receive it
  // Then do the opposite for the same collision
  // i.e. send right first then send to the left

  // when receiving copy data into results


    COLLIDE_Destroy(c->collide_grid);
}



void ParFUM_Collide_Destroy(ParFUM_collider *c){
  // if we setup the collision library in init, then we should clean it up here.
  c->collide_grid=NULL;
}

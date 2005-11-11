/* 
ParFUM Collision Interface File

A few outstanding questions:

Is the use of an element based attribute to store any needed collision data a good thing?
Perhaps we should just use the user data attributes for the element. This may require there 
to be consequtive user data attributes. I.e. no FEM_DATA+5,FEM_DATA+82, without those inbetween.
Do we need to transmit nodal data for each element?
Does the user need anything beyond just some data attributes for the one remote element which is 
colliding locally?


Author: Isaac Dooley 11-09-2005
*/


#ifndef _CHARM_FEM_COLLIDE_H
#define _CHARM_FEM_COLLIDE_H


#ifdef __cplusplus
extern "C" {
#endif


  /* ParFUM_Collide_init() will initialize the collision library. 
     It should be called once in driver after mesh has been loaded.
     
     dimension should reflect the number of coordinates associated 
     with a node. This cannot exceed 3 with the current Collision
     Library. The user's nodal coordinates must be registered as a 
     particular attribute in order to determine the optimal grid sizing.

     Algorithm:
       Determine Grid Sizing
       Call COLLIDE_Init()
     
  */   
  collide_t ParFUM_Collide_Init(int dimension);


  /* ParFUM_Collide() will create bounding boxes for each element in the local mesh chunk.
     It will then collide these bounding boxes with those both locally and remotely.
     It should be called at each timestep for which collisions are being tested.
    
     Algorithm: 
       Create Bounding boxes for all valid elements, and priority array
       Call COLLIDE_Boxes_prio()
       return the number of collisions which involve a local element
  */  
  int ParFUM_Collide(collide_t c);


  /* ParFUM_Collide_GetCollisions() is used to get the data for any remote elements which 
     It should be called after Collide even if ParFUM_Collide returned 0

     The data it returns will be double precision values associated with the
     element attribute ParFUM_COLLISION_DATA

     results should be an array allocated by the user with length equal to the number of 
     collisions times the amount of space needed for each item in the ParFUM_COLLISION_DATA 
     attribute
          
     Algorithm: 



  */  
  void ParFUM_Collide_GetCollisions(collide_t c, void* results);


}

#endif

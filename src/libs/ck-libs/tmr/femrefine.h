/** 
New interface for tmr to be used along with FEM. It automatically updates the FEM mesh while
refining. It uses the new registration api in FEM and its use can be found in pgm.C.

created by Sayantan Chakravorty, 03/22/2004
*/

#ifndef _UIUC_CHARM_FEMREFINE_IMPL_H
#define _UIUC_CHARM_FEMREFINE_IMPL_H

typedef void (*repeat_split_fn)(void *data);

#ifdef __cplusplus
extern "C" {
#endif

/***
 * Create a new refinement object for this virtual processor.
 * Must be called exactly once at startup.
 */
 
void FEM_REFINE2D_Init();


/*
 * Use this call to set up the refinement framework for
 * a new mesh. This must be called before the split call.
 meshID - mesh to refine
 nodeID - FEM_NODE +t .. which type of nodes to refine
 elemID - FEM_ELEM +t .. which type of element to refine
*/

void FEM_REFINE2D_Newmesh(int meshID,int nodeID,int elemID);

/* This function refines a mesh, to the desired degree and updates the FEM mesh.
	 Arguments
	 	meshID - which mesh
		nodeID - FEM_NODE+t .. which type of nodes
		coord  - the cordinate of each node in an array of doubles
						 node i has its x,y cordinates in coord[2*i] and coord[2*i+1]
		elemID - FEM_ELEM+t .. type of element
		desiredAreas - desiredArea[i] gives the desired area of element i.
*/

void FEM_REFINE2D_Split(int meshID,int nodeID,double *coord,int elemID,double *desiredAreas,int sparseID=-1);


#ifdef __cplusplus
};
#endif


#endif

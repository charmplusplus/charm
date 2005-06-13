/** 
New interface for tmr to be used along with FEM. It automatically updates the FEM mesh while
refining. It uses the new registration api in FEM and its use can be found in pgm.C.

created by Sayantan Chakravorty, 03/22/2004
*/

#ifndef _UIUC_CHARM_FEMREFINE_IMPL_H
#define _UIUC_CHARM_FEMREFINE_IMPL_H

typedef void (*repeat_split_fn)(void *data);
/*system attribute of an entity which marks whether a node is valid or not
  node and element width = 1 and type = FEM_BYTE
*/
#define FEM_VALID FEM_ATTRIB_TAG_MAX-1 

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
 nodeBoundary - nodeBoundary marks whether nodes have boundary flags
*/

void FEM_REFINE2D_Newmesh(int meshID,int nodeID,int elemID,int nodeBoundary=0);

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

void FEM_REFINE2D_Coarsen(int meshID,int nodeID,double *coord,int elemID,double *desiredAreas,int sparseID=-1);


#ifdef __cplusplus
};
#endif


#endif

/*Charm++ Mesh Refinement Framework:
C++ implementation file

Orion Sky Lawlor, olawlor@acm.org, 4/8/2002
Modified by Terry Wilmarth, wilmarth@cse.uiuc.edu, 4/16/2002
*/
#ifndef _UIUC_CHARM_REFINE_IMPL_H
#define _UIUC_CHARM_REFINE_IMPL_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Create a new refinement object for this virtual processor.
 * Must be called exactly once at startup.
 */
void REFINE2D_Init(void);

/**
 * Push a new mesh into the refinement system.  This is the first
 * call user programs make into the refinement system.  This call
 * need *not* be repeated when the refinement system changes the mesh;
 * only when the user changes the mesh (e.g., to coarsen it).
 *
 * Conn is row-major, and maps an element number to three node numbers.
 * Hence conn[i*3+j] gives the local number of node j of element i.
 * Because of shared nodes, every node of every local element will be local.
 * Ghost elements may not have the complete set of nodes-- some of their
 * nodes may have the invalid number -1.
 * 
 * Elements with numbers between 0 and nEl-1 (inclusive) are local.
 * Elements with numbers between nEl and nGhost-1 (inclusive) are "ghosts"--
 * elements that are actually local on another processor.  There are
 * guaranteed to be enough ghosts that every local element's non-boundary
 * edge will face a ghost element.
 *
 * gid maps an element number to a chunk number and local number on that
 * chunk.  These are stored at gid[i*2+0] (chunk number) and gid[i*2+1]
 * (local number).
 */
void REFINE2D_NewMesh(int nEl,int nGhost,const int *conn,const int *gid);

/**
 * Refine the mesh so each element has, at most, the given desired area.
 * 
 * Coord gives the (x,y) coordinates of nodes 0..nNode-1.
 * coord[2*j+0] is the x coordinate of node j
 * coord[2*j+1] is the y coordinate of node j
 *
 * desiredArea[i] gives the desired area of element i.  nEl must be 
 * equal to the "nEl" value passed in via REFINE_NewMesh plus the number
 * of "split" calls received during earlier refinements.
 */
void REFINE2D_Split(int nNode,double *coord,int nEl,double *desiredArea);

/**
 * Get the number of split triangles.
 */
int REFINE2D_Get_Split_Length(void);

  /**
   * Return the indices (tri) and edge number (edge) of each split triangle.
   *
   * For our purposes, edges are numbered 0 (connecting nodes 0 and 1), 
   * 1 (connecting 1 and 2), and 2 (connecting 2 and 0).
   * 
   * Taking as A and B the (triangle-order) nodes of the splitting edge:
   *
   *                     C                      C                 
   *                    / \                    /|\                  
   *                   /   \                  / | \                 
   *                  /     \      =>        /  |  \                
   *                 /       \              /   |   \               
   *                /         \            /old | new\            
   *               B --------- A          B --- D --- A         
   *
   *   The original triangle's node A should be replaced by D;
   * while a new triangle should be inserted with nodes CAD.
   *
   *   The new node D's location should equal A*(1-frac)+B*frac.
   * For a simple splitter, frac will always be 0.5.
   *
   *   If nodes A and B are shared with some other processor,
   * that processor will also receive a "split" call for the
   * same edge.  If nodes A and B are shared by some other local
   * triangle, that triangle will immediately receive a "split" call
   * for the same edge.  
   *
   * Client's responsibilities:
   *   -Add the new node D.  Since both sides of a shared local edge
   *      will receive a "split" call, you must ensure the node is
   *      not added twice.
   *   -Update connectivity for source triangle
   *   -Add new triangle. 
   */

void REFINE2D_Get_Splits(int nSplits,int *tri,int *edge,int *movingNode,
	double *frac);

/**
 * Check to make sure our connectivity and the refine connectivity agree.
 */
void REFINE2D_Check(int nEle,const int *conn,int nNodes);

#ifdef __cplusplus
};
#endif

#endif



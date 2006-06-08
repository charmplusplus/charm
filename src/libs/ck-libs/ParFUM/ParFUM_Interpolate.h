/* File: interpolate.h
 * Authors: Terry Wilmarth, Nilesh Choudhury
 * 
 */

#ifndef __ParFUM_INTERPOLATE_H
#define __ParFUM_INTERPOLATE_H

class femMeshModify;

///Interface for solution transfer operations
/** This is an interface for specifying solution transfer operations
    for small mesh modification operations. It provides overridable
    defaults for node and element interpolations.  These functions are passed to
    FEM_add_element and FEM_add_node operations, along with a simple argument
    struct - TLW
*/
class FEM_Interpolate {
  ///cross-pointer to the Mesh object for this chunk
  FEM_Mesh *theMesh;
  ///cross-pointer to the femMeshModify object for this chunk
  femMeshModify *theMod;

 public:
  ///The input information for a nodal interpolate operation
  typedef struct {
    int n;
    int nodes[8];
    int dim;
    int nNbrs;
    double coord[3];
    double frac;
    bool addNode;
  } NodalArgs;
  ///The input formation for an element interpolate(copy) operation
  typedef struct {
    int e;
    int oldElement;
    int elType;
  } ElementArgs;
  ///The node interpolate function type
  typedef void (* FEM_InterpolateNodeFn)(NodalArgs, FEM_Mesh *);
  ///The element interpolate function type
  typedef void (* FEM_InterpolateElementFn)(ElementArgs);

  ///Node interpolate functions along edge, face, element or node copy
  FEM_InterpolateNodeFn nodeEdgeFnPtr, nodeFaceFnPtr, nodeElementFnPtr, nodeCopyFnPtr;
  ///Element interpolate functions for element copy and elemToNode copy
  FEM_InterpolateElementFn elemCopyFnPtr, elemNodeFnPtr;

 public:
  /// Basic Constructor
  FEM_Interpolate() {
    nodeEdgeFnPtr = nodeFaceFnPtr = nodeElementFnPtr = nodeCopyFnPtr = NULL;
    elemCopyFnPtr = elemNodeFnPtr = NULL;
    theMesh = NULL;
    theMod = NULL;
  }
  ///Initialize only the mesh object of this chunk (constructor)
  FEM_Interpolate(FEM_Mesh *m) {
    nodeEdgeFnPtr = nodeFaceFnPtr = nodeElementFnPtr = nodeCopyFnPtr = NULL;
    elemCopyFnPtr = elemNodeFnPtr = NULL;
    theMesh = m;
    theMod = NULL;
  }
  ///Initialize both the mesh and femMeshModify object of this chunk (constructor)
  FEM_Interpolate(FEM_Mesh *m, femMeshModify *fm) {
    nodeEdgeFnPtr = nodeFaceFnPtr = nodeElementFnPtr = nodeCopyFnPtr = NULL;
    elemCopyFnPtr = elemNodeFnPtr = NULL;
    theMesh = m;
    theMod = fm;
  }
  ///Initialize only the femMeshModify object of this chunk (constructor)
  FEM_Interpolate(femMeshModify *fm) {
    nodeEdgeFnPtr = nodeFaceFnPtr = nodeElementFnPtr = nodeCopyFnPtr = NULL;
    elemCopyFnPtr = elemNodeFnPtr = NULL;
    theMesh = NULL;
    theMod = fm;
  }
  ///Pup operation for this object
  void pup(PUP::er &p) {
    //p|theMesh;
    //p|theMod;
  }
  ///Initialize the mesh object for this chunk
  void FEM_InterpolateSetMesh(FEM_Mesh *m) { theMesh = m; }
  
  /** Methods to set and reset interpolate functions on the fly; these will be
      used to override defaults by Fortan code, and can also be set and reset
      or using temporary special-purpose interpolation functions */
  ///Set interpolate function for a node on an edge
  void FEM_SetInterpolateNodeEdgeFnPtr(FEM_InterpolateNodeFn fnPtr) {
    nodeEdgeFnPtr = fnPtr;
  }
  ///Set interpolate function for a node on a face
  void FEM_SetInterpolateNodeFaceFnPtr(FEM_InterpolateNodeFn fnPtr) {
    nodeFaceFnPtr = fnPtr;
  }
  ///Set interpolate function for a node from an element
  void FEM_SetInterpolateNodeElementFnPtr(FEM_InterpolateNodeFn fnPtr) {
    nodeElementFnPtr = fnPtr;
  }
  ///Set interpolate function for a node by copying from another node
  void FEM_SetInterpolateCopyAttributesFnPtr(FEM_InterpolateNodeFn fnPtr) {
    nodeCopyFnPtr = fnPtr;
  }
  ///Set interpolate function for an element by copying from another element
  void FEM_SetInterpolateElementCopyFnPtr(FEM_InterpolateElementFn fnPtr) {
    elemCopyFnPtr = fnPtr;
  }
  ///Set interpolate function for an element from a node
  void FEM_SetInterpolateElementNodeFnPtr(FEM_InterpolateElementFn fnPtr) {
    elemNodeFnPtr = fnPtr;
  }
  ///Reset function pointer
  void FEM_ResetInterpolateNodeEdgeFnPtr() { nodeEdgeFnPtr = NULL; }
  ///Reset function pointer
  void FEM_ResetInterpolateNodeFaceFnPtr() { nodeFaceFnPtr = NULL; }
  ///Reset function pointer
  void FEM_ResetInterpolateNodeElementFnPtr() { nodeElementFnPtr = NULL; }
  ///Reset function pointer
  void FEM_ResetInterpolateElementCopyFnPtr() { elemCopyFnPtr = NULL; }  
  ///Reset function pointer
  void FEM_ResetInterpolateElementNodeFnPtr() { elemNodeFnPtr = NULL; }
  ///Reset function pointer
  void FEM_ReetInterpolateCopyAttributesFnPtr() { nodeCopyFnPtr = NULL; }

  // Nodal data
  /// A node is added on an edge; interpolate from neighboring nodes
  virtual void FEM_InterpolateNodeOnEdge(NodalArgs args);
  /// A node is added on an face; interpolate from nodes of face
  virtual void FEM_InterpolateNodeOnFace(NodalArgs args);
  /// A node is added inside a volume; interpolate from nodes of element
  virtual void FEM_InterpolateNodeInElement(NodalArgs args);

  /// An element added is completely encapsulated by element to be removed
  virtual void FEM_InterpolateElementCopy(ElementArgs args);
  /// An element is added and derives data from its nodes
  virtual void FEM_InterpolateElementFromNodes(ElementArgs args);
  /// Store data of an element temporarily on all nodes
  virtual void FEM_InterpolateElementToNodes(int e);

  ///Copy the data from one node to another
  virtual void FEM_InterpolateCopyAttributes(int oldnode, int newnode);
};

// End interpolate.h

#endif

/* A node is added on an edge; interpolate from neighboring nodes; this uses n,
   nodes[2], dim and frac. Frac is between 0.0 and 1.0 and weights nodes[0]; 
   i.e. if frac=1.0, n gets a copy of nodes[0]'s data, and nodes[0]'s coords */
void FEM_Interpolate::FEM_InterpolateNodeOnEdge(NodalArgs args)
{
  if (nodeEdgeFnPtr) { // default is overridden
    nodeEdgeFnPtr(args);
    return;
  }
  // do default interpolation
  // DEFAULT BEHAVIOR: Only new node's coordinates calculated as weighted avg
  // of other node coords
}

/* A node is added on an face; interpolate from nodes of face; this uses n, 
   nodes[3] or nodes[4] depending on element type, dim and coord */
void FEM_Interpolate::FEM_InterpolateNodeOnFace(NodalArgs args)
{
  if (nodeFaceFnPtr) { // default is overridden
    nodeFaceFnPtr(args);
    return;
  }
  // do default interpolation
  // DEFAULT BEHAVIOR: Only new node's coordinates calculated as weighted avg
  // of other node coords
}

/* A node is added inside a volume; interpolate from nodes of element; this
   uses n, nodes[4] or more, and coord */
void FEM_Interpolate::FEM_InterpolateNodeInElement(NodalArgs args)
{
  if (nodeElementFnPtr) { // default is overridden
    nodeElementFnPtr(args);
    return;
  }
  // do default interpolation
  // DEFAULT BEHAVIOR: Only new node's coordinates calculated as weighted avg
  // of other node coords
}

/* An element added is completely encapsulated by element to be removed; for 
   example, edge bisect two elements replace the old element, and we simply 
   copy the data of the old element to both new elements */
void FEM_Interpolate::FEM_InterpolateElementCopy(ElementArgs args)
{
  if (elemCopyFnPtr) { // default is overridden
    elemCopyFnPtr(args);
    return;
  }
  // do default interpolation
  // DEFAULT BEHAVIOR: COPY ALL ELEMENT DATA
}

/* An element is added and derives data from its nodes; assumes relevant data 
   was copied to the appropriate nodes prior to this operation; see the utility
   function below */
void FEM_Interpolate::FEM_InterpolateElementFromNodes(ElementArgs args)
{
  if (elemNodeFnPtr) { // default is overridden
    elemNodeFnPtr(args);
    return;
  }
  // do default interpolation
  // DEFAULT BEHAVIOR: NO ELEMENT DATA
}

/* Store data of an element temporarily on all nodes; this data is used later 
   to derive an element's data */
void FEM_Interpolate::FEM_InterpolateElementToNodes(int e)
{
  // DEFAULT BEHAVIOR: NO ELEMENT DATA
  // This function intentionally left blank; derived classes may define it, or
  // user is responsible for transferring element data to nodes on their own
}

/**
   @file
   @brief A ParFUM "Tops" compatibility layer API Definition
   
   @author Isaac Dooley

   ParFUM-TOPS provides a Tops-like API for ParFUM.

\note \code
Sample usage:

  // Reads nodes (id, x, y, z)
  for (i = 0; i < nn; i++)
  {
      double x, y, z;
      TopNode node;
      NodeAtt*  node_data;
      if (fscanf(fp,"%d, %lf, %lf, %lf",&id, &x, &y, &z) != 4) {
          fprintf(stderr,"Invalid format for nodes.\n");
          exit(1);
          }
      // Adds node to the model
      node = topModel_InsertNode (model, x, y, z);
      topNode_SetId (model, node, id);
      node_data = (NodeAtt*) malloc(sizeof(NodeAtt));
      assert(node_data);
      initNodeAtt(node_data);
      node_data->material.E = material.E;
      node_data->material.v = material.v;
      node_data->material.p = material.p;
      node_data->bc = 0;
      topNode_SetAttrib (model, node, node_data);
  }




	 TopNodeItr* itr = topModel_CreateNodeItr(m);
	  int node_count=0;
	  for(topNodeItr_Begin(itr);topNodeItr_IsValid(itr);topNodeItr_Next(itr)){
		node_count++;
		TopNode node = topNodeItr_GetCurr(itr);
		NodeAtt* na = topNode_GetAttrib(m,node);
		print_node_attribute(myId, na);
	  }
	  printf("vp %d: node_count = %d\n", myId, node_count);


\endcode

*/


#include <ParFUM.h>
#include <ParFUM_internals.h>


/** A tops model is roughly equivalent to a ParFUM FEM_Mesh object */
typedef FEM_Mesh TopModel;

/** Tops uses some bit patterns for these, but we just use TopNode as a signed value to represent the corresponding ParFUM node. A non-negative value is a local node, while a negative value is a ghost. */
typedef unsigned TopNode;
/** See notes for ::TopNode */
typedef unsigned TopElement;

typedef void NodeAtt;
/** See notes for ::NodeAtt */
typedef void ElemAtt;

/** Node Iterator */
class TopNodeItr{
public:
  /** The signed index used to refer to a ParFUM Element. Non-negatives are ghosts*/ 
  int parfum_index;
  /** The associated model */
  TopModel *model;
};

/** Element Iterator. See notes for class TopNodeItr */
class TopElemItr{
public:
  int parfum_index;
    TopModel *model;
};

/** an opaque id for top entities */
typedef int TopID;

/** an enumeration of supported element types */
typedef int TopElemType;


/** 
Create and access a Tops model. Only call from Init 
Currently only one model can be created. To extend, each model must just reference a different FEM_Mesh object
*/
TopModel* topModel_Create_Init(int elem_attr_sz, int node_attr_sz);

/** Create and access a Tops model. Only call from Driver */
TopModel* topModel_Create_Driver(int elem_attr_sz, int node_attr_sz);

/** Insert a node */
TopNode topModel_InsertNode(TopModel*, double x, double y, double z);

/** Set id of a node */
void topNode_SetId(TopModel*, TopNode, TopID id);

/** Set attribute of a node */
void topNode_SetAttrib(TopModel*, TopNode, NodeAtt*);

/** Insert an element */
TopElement topModel_InsertElem(TopModel*, TopElemType, TopNode*);

/** Set id of an element */
void topElement_SetId(TopModel*, TopElement, TopID id);

/** Set attribute of an element */
void topElement_SetAttrib(TopModel*, TopElement, ElemAtt*);

/** Get node via id */
TopNode topModel_GetNodeAtId(TopModel*,TopID);

/** Get elem via id */
TopElement topModel_GetElemAtId(TopModel*,TopID);

/** Get nodal attribute */
NodeAtt* topNode_GetAttrib(TopModel*, TopNode);

/** Create Iterator for nodes */
TopNodeItr*  topModel_CreateNodeItr(TopModel*);

/** Destroy Iterator */
void topNodeItr_Destroy(TopNodeItr*);

/** Initialize Iterator */
void topNodeItr_Begin(TopNodeItr*);

/** Determine if Iterator is valid or if it has iterated past last Node */
bool topNodeItr_IsValid(TopNodeItr*);

/** Increment iterator */
void topNodeItr_Next(TopNodeItr*);

/** Get TopNode associated with the iterator */
TopNode topNodeItr_GetCurr(TopNodeItr*);


/** Create Iterator for elements */
TopElemItr*  topModel_CreateElemItr(TopModel*);

/** Destroy Iterator */
void topElemItr_Destroy(TopElemItr*);

/** Initialize Iterator */
void topElemItr_Begin(TopElemItr*);

/** Determine if Iterator is valid or if it has iterated past last Element */
bool topElemItr_IsValid(TopElemItr*);

/** Increment iterator */
void topElemItr_Next(TopElemItr*);

/** Get TopElement associated with the iterator */
TopElement topElemItr_GetCurr(TopElemItr*);




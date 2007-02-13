/**
*	A ParFUM TOPS compatibility layer
*
*      Author: Isaac Dooley



LB & PUPing


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


*/


#include <ParFUM.h>
#include <ParFUM_internals.h>



/** some abstract data containers which will need to make sense at some point. */
class TopModel{
public:
    FEM_Mesh *mesh;

	// add a hash table for the attribute pointers here
	// index into here with an int that is possibly stored in ParFUM int attribute

	TopModel(){
		mesh=FEM_Mesh_default_read();  // fix this to get the right FEM_Mesh instead of an int
	}

};

typedef unsigned TopNode;
typedef unsigned TopElement;
class NodeAtt{};
class ElemAtt{};

/** Iterators */
class TopNodeItr{
public:
    int parfum_nodal_index;
    TopModel *model;
};

class TopElemItr{};

/** an opaque id for top entities */
typedef int TopID;

/** an enumeration of supported element types */
typedef int TopElemType;


/** TOPS functions we need to support */

/** Create a new model(essentially a mesh) */
TopModel* topModel_Create();

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

/** C-like Iterator for nodes */
TopNodeItr*  topModel_CreateNodeItr(TopModel*);
void topNodeItr_Destroy(TopNodeItr*);
void topNodeItr_Begin(TopNodeItr*);
bool topNodeItr_IsValid(TopNodeItr*);
void topNodeItr_next(TopNodeItr*);
TopNode topNodeItr_GetCurr(TopNodeItr*);

/** C-like Iterator for elements */
TopElemItr*  topModel_CreateElemItr(TopModel*);
void topElemItr_Destroy(TopElemItr*);
void topElemItr_Begin(TopElemItr*);
bool topElemItr_IsValid(TopElemItr*);
void topElemItr_next(TopElemItr*);
TopElement topElemItr_GetCurr(TopElemItr*);




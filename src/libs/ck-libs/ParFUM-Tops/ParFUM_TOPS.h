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


typedef FEM_Mesh TopModel;

typedef unsigned TopNode;
typedef unsigned TopElement;

class NodeAtt;
class ElemAtt;

/** Iterators */
class TopNodeItr{
public:
    int parfum_index;
    TopModel *model;
};

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

/** C-like Iterator for nodes */
TopNodeItr*  topModel_CreateNodeItr(TopModel*);
void topNodeItr_Destroy(TopNodeItr*);
void topNodeItr_Begin(TopNodeItr*);
bool topNodeItr_IsValid(TopNodeItr*);
void topNodeItr_Next(TopNodeItr*);
TopNode topNodeItr_GetCurr(TopNodeItr*);

/** C-like Iterator for elements */
TopElemItr*  topModel_CreateElemItr(TopModel*);
void topElemItr_Destroy(TopElemItr*);
void topElemItr_Begin(TopElemItr*);
bool topElemItr_IsValid(TopElemItr*);
void topElemItr_Next(TopElemItr*);
TopElement topElemItr_GetCurr(TopElemItr*);




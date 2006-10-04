/**  
*	A ParFUM TOPS compatibility layer
*	
*      Author: Isaac Dooley 
*/


#include <ParFUM.h>
#include <ParFUM_internals.h>


/** some abstract data containers which will need to make sense at some point. */
class TopModel{
public:
    FEM_Mesh *mesh;
};

class TopNode{};
class NodeAtt{};
class ElemAtt{};
class TopElement{};

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




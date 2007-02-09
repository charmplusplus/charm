/**
*	A ParFUM TOPS compatibility layer
*
*      Author: Isaac Dooley
*/


#include <ParFUM_TOPS.h>


TopModel* topModel_Create(){
    return new TopModel;
    // We really need to support multiple meshes for this to work

	// Setup tables for attributes for the new model

}

TopNode topModel_InsertNode(TopModel*, double x, double y, double z){
// TODO : insert a node here
TopNode a;
return a;
}


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
TopNodeItr*  topModel_CreateNodeItr(TopModel* model){
    TopNodeItr *itr = new TopNodeItr;
    itr->model = model;
    return itr;
}

void topNodeItr_Destroy(TopNodeItr* itr){
    delete itr;
}

void topNodeItr_Begin(TopNodeItr* itr){
    itr->parfum_nodal_index = 0;
}

bool topNodeItr_IsValid(TopNodeItr*itr){
     return itr->model->mesh->node.is_valid(itr->parfum_nodal_index);
}

void topNodeItr_next(TopNodeItr* itr){

    if(!topNodeItr_IsValid(itr))
        return;

    // advance index until we hit a valid index
    itr->parfum_nodal_index++;

    if(itr->parfum_nodal_index > 0) {// local nodes

        while ((! itr->model->mesh->node.is_valid(itr->parfum_nodal_index)) &&
                  (itr->parfum_nodal_index<itr->model->mesh->node.size()))
        {
            itr->parfum_nodal_index++;
        }

        if(itr->model->mesh->node.is_valid(itr->parfum_nodal_index)) {
            return;
        } else {
            // cycle to most negative index possible for ghosts
            itr->parfum_nodal_index = FEM_To_ghost_index(itr->model->mesh->node.ghost->size());
        }
    }

    // just go through ghost nodes

    while ( (! itr->model->mesh->node.ghost->
                is_valid(FEM_To_ghost_index(itr->parfum_nodal_index)))
                &&
                itr->parfum_nodal_index<0)
    {
        itr->parfum_nodal_index++;
    }

    if(itr->parfum_nodal_index==0){
        itr->parfum_nodal_index = itr->model->mesh->node.size()+1000; // way past the end
    }

}

TopNode topNodeItr_GetCurr(TopNodeItr*itr){
    // TODO lookup data associated with this iterator
    TopNode a;
return a;
}

/** C-like Iterator for elements */
TopElemItr*  topModel_CreateElemItr(TopModel*);
void topElemItr_Destroy(TopElemItr*);
void topElemItr_Begin(TopElemItr*);
bool topElemItr_IsValid(TopElemItr*);
void topElemItr_next(TopElemItr*);
TopElement topElemItr_GetCurr(TopElemItr*);



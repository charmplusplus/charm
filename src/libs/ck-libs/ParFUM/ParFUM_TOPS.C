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
    // should return true if there is a node at the current index 
    // or if there are nodes past this index
    // or if there are any ghost nodes

    if (itr->parfum_nodal_index > itr->model->mesh->node.size())
        return false;

    return true;
}

void topNodeItr_next(TopNodeItr* itr){

    if(!topNodeItr_IsValid(itr))
        return;

    // advance index until we hit a valid index
    itr->parfum_nodal_index++;

    if(itr >= 0) {// local nodes

        while ((! itr->model->mesh->node.is_valid(itr->parfum_nodal_index)) && 
                  (itr->parfum_nodal_index<itr->model->mesh->node.size()))
        {
            itr->parfum_nodal_index++;
        }

        // TODO next go through the ghosts if we didn't get to a valid local node

    }
    else { // just go through ghost nodes


    }


}

TopNode topNodeItr_GetCurr(TopNodeItr*itr){
    // TODO somehow need to get back to a TopNode
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



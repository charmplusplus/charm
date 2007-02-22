#include "ParFUM_TOPS.h"

/**************************************************************************
 *     Iterator for nodes 
 */

TopNodeItr*  topModel_CreateNodeItr(TopModel* model){
    TopNodeItr *itr = new TopNodeItr;
    itr->model = model;
    return itr;
}

void topNodeItr_Destroy(TopNodeItr* itr){
    delete itr;
}

void topNodeItr_Begin(TopNodeItr* itr){
    itr->parfum_index = 0;
}

bool topNodeItr_IsValid(TopNodeItr*itr){
     return itr->model->node.is_valid_any_idx(itr->parfum_index);
}

void topNodeItr_Next(TopNodeItr* itr){

    if(!topNodeItr_IsValid(itr))
        return;

    // advance index until we hit a valid index
    itr->parfum_index++;

    if(itr->parfum_index > 0) {// local nodes

        while ((! itr->model->node.is_valid_any_idx(itr->parfum_index)) &&
                  (itr->parfum_index<itr->model->node.size()))
        {
            itr->parfum_index++;
        }

        if(itr->model->node.is_valid_any_idx(itr->parfum_index)) {
            return;
        } else {
            // cycle to most negative index possible for ghosts
		  printf("Node iterator switched to ghosts\n");
            itr->parfum_index = FEM_To_ghost_index(itr->model->node.ghost->size());
        }
    }

    // just go through ghost nodes
    
    while ( (! itr->model->node.ghost->
                is_valid_any_idx(FEM_To_ghost_index(itr->parfum_index)))
                &&
                itr->parfum_index<0)
    {
        itr->parfum_index++;
    }

    if(itr->parfum_index==0){
        itr->parfum_index = itr->model->node.size()+1000; // way past the end
    }

}

TopNode topNodeItr_GetCurr(TopNodeItr*itr){
	return itr->parfum_index;
}


/**************************************************************************
 *     Iterator for elements
 */

TopElemItr*  topModel_CreateElemItr(TopModel* model){
    TopElemItr *itr = new TopElemItr;
    itr->model = model;
    return itr;
}

void topElemItr_Destroy(TopElemItr* itr){
    delete itr;
}

void topElemItr_Begin(TopElemItr* itr){
    itr->parfum_index = 0;
}

bool topElemItr_IsValid(TopElemItr*itr){
     return itr->model->elem[0].is_valid_any_idx(itr->parfum_index);
}

void topElemItr_Next(TopElemItr* itr){

    if(!topElemItr_IsValid(itr))
        return;

    // advance index until we hit a valid index
    itr->parfum_index++;

    if(itr->parfum_index > 0) {// non-ghosts

        while ((! itr->model->elem[0].is_valid_any_idx(itr->parfum_index)) &&
                  (itr->parfum_index<itr->model->elem[0].size()))
        {
            itr->parfum_index++;
        }

        if(itr->model->elem[0].is_valid_any_idx(itr->parfum_index)) {
            return;
        } else {
            // cycle to most negative index possible for ghosts
		  printf("Elem iterator switched to ghosts\n");
            itr->parfum_index = FEM_To_ghost_index(itr->model->elem[0].ghost->size());
        }
    }

    // just go through ghosts    
    while ( (! itr->model->elem[0].ghost->
                is_valid_any_idx(FEM_To_ghost_index(itr->parfum_index)))
                &&
                itr->parfum_index<0)
    {
        itr->parfum_index++;
    }

    if(itr->parfum_index==0){
        itr->parfum_index = itr->model->elem[0].size()+1000; // way past the end
    }

}

TopNode topElemItr_GetCurr(TopElemItr*itr){
	return itr->parfum_index;
}

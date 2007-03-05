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
  if(itr->model->mesh->node.ghost != NULL){
	itr->parfum_index =  FEM_To_ghost_index(itr->model->mesh->node.ghost->size());
  }
  else{
	itr->parfum_index =  0;
  }

  // Make sure we start with a valid one:
  while((!itr->model->mesh->node.is_valid_any_idx(itr->parfum_index)) &&
		(itr->parfum_index < itr->model->mesh->node.size()))
	itr->parfum_index++;

  if(itr->parfum_index==itr->model->mesh->node.size()){
	itr->parfum_index = itr->model->mesh->node.size()+1000; // way past the end
  }

#ifdef PTOPS_ITERATOR_PRINT
  CkPrintf("Initializing Node Iterator to %d\n", itr->parfum_index);
#endif
}

bool topNodeItr_IsValid(TopNodeItr*itr){
  	return itr->model->mesh->node.is_valid_any_idx(itr->parfum_index);
}

void topNodeItr_Next(TopNodeItr* itr){
  CkAssert(topNodeItr_IsValid(itr));

  // advance index until we hit a valid index
  itr->parfum_index++;

  while((!itr->model->mesh->node.is_valid_any_idx(itr->parfum_index)) &&
		(itr->parfum_index < itr->model->mesh->node.size()))
	itr->parfum_index++;

  if(itr->parfum_index==itr->model->mesh->node.size()){
	itr->parfum_index = itr->model->mesh->node.size()+1000; // way past the end
  }

#ifdef PTOPS_ITERATOR_PRINT
  CkPrintf("Advancing Node Iterator to %d\n", itr->parfum_index);
#endif
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
  if(itr->model->mesh->elem[0].ghost != NULL){
	itr->parfum_index =  FEM_To_ghost_index(itr->model->mesh->elem[0].ghost->size());
  }
  else{
	itr->parfum_index =  0;
  }

  // Make sure we start with a valid one:
  while((!itr->model->mesh->elem[0].is_valid_any_idx(itr->parfum_index)) &&
		(itr->parfum_index < itr->model->mesh->elem[0].size()))
	itr->parfum_index++;

  if(itr->parfum_index==itr->model->mesh->elem[0].size()){
	itr->parfum_index = itr->model->mesh->elem[0].size()+1000; // way past the end
  }

#ifdef PTOPS_ITERATOR_PRINT
  CkPrintf("Initializing Elem[0] Iterator to %d\n", itr->parfum_index);
#endif

}

bool topElemItr_IsValid(TopElemItr*itr){
  return itr->model->mesh->elem[0].is_valid_any_idx(itr->parfum_index);
}

void topElemItr_Next(TopElemItr* itr){
  CkAssert(topElemItr_IsValid(itr));

  // advance index until we hit a valid index
  itr->parfum_index++;

  while((!itr->model->mesh->elem[0].is_valid_any_idx(itr->parfum_index)) &&
		(itr->parfum_index < itr->model->mesh->elem[0].size()))
	itr->parfum_index++;


  if(itr->parfum_index==itr->model->mesh->elem[0].size()){
	itr->parfum_index = itr->model->mesh->elem[0].size()+1000; // way past the end
  }

#ifdef PTOPS_ITERATOR_PRINT
  CkPrintf("Advancing Elem Iterator to %d\n", itr->parfum_index);
#endif
}


TopNode topElemItr_GetCurr(TopElemItr*itr){
	return itr->parfum_index;
}

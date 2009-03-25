#include "ParFUM_TOPS.h"


#ifndef INLINE_ITERATORS

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
	CkAssert(topNodeItr_IsValid(itr));
	return itr->parfum_index;
}


/**************************************************************************
 *     Iterator for elements
 * 
 *  TODO: Iterate through both cohesives & bulk elements ?
 * 
 */

TopElemItr*  topModel_CreateElemItr(TopModel* model){
    TopElemItr *itr = new TopElemItr;
    itr->model = model;
    itr->type = TOP_ELEMENT_TET4;
    return itr;
}

void topElemItr_Destroy(TopElemItr* itr){
    delete itr;
}

void topElemItr_Begin(TopElemItr* itr){
	itr->done = false;

	if(itr->model->mesh->elem[itr->type].ghost != NULL){
		itr->parfum_index =  FEM_To_ghost_index(itr->model->mesh->elem[itr->type].ghost->size());
	}
	else{
		itr->parfum_index =  0;
	}

	// Make sure we start with a valid one:
	while((!itr->model->mesh->elem[itr->type].is_valid_any_idx(itr->parfum_index)) &&
			(itr->parfum_index < itr->model->mesh->elem[itr->type].size()))
		itr->parfum_index++;

	if(itr->parfum_index==itr->model->mesh->elem[itr->type].size()){
		itr->done = true;
	}

#ifdef PTOPS_ITERATOR_PRINT
	CkPrintf("Initializing elem[itr->type] Iterator to %d\n", itr->parfum_index);
#endif

}

bool topElemItr_IsValid(TopElemItr*itr){
  return ! itr->done;
}

void topElemItr_Next(TopElemItr* itr){
	CkAssert(topElemItr_IsValid(itr));

	// advance index until we hit a valid index
	itr->parfum_index++;

	while((!itr->model->mesh->elem[TOP_ELEMENT_TET4].is_valid_any_idx(itr->parfum_index)) &&
			(itr->parfum_index < itr->model->mesh->elem[TOP_ELEMENT_TET4].size()))
		itr->parfum_index++;


	if(itr->parfum_index==itr->model->mesh->elem[TOP_ELEMENT_TET4].size()){
		itr->done = true;
	}

#ifdef PTOPS_ITERATOR_PRINT
	CkPrintf("Advancing Elem Iterator to %d\n", itr->parfum_index);
#endif
}


TopElement topElemItr_GetCurr(TopElemItr*itr){	
	CkAssert(topElemItr_IsValid(itr));
	TopElement e;
	e.id = itr->parfum_index; 
	e.type = itr->type;
	return e;
}



/**************************************************************************
 *     Iterator for elements adjacent to a node
 * 
 *  TODO: Iterate through both cohesives & bulk elements ?
 * 
 */



TopNodeElemItr* topModel_CreateNodeElemItr (TopModel* model, TopNode n){
	TopNodeElemItr *itr = new TopNodeElemItr;
	
	itr->model = model;
	itr->node = n;
	itr->current_index=0;
	
	if( itr->model->mesh->node.is_valid_any_idx(n) ){
		itr->numAdjElem = model->mesh->n2e_getLength(n);	
	} else {
		itr->numAdjElem = -1;
	}
	
	return itr;
}


bool topNodeElemItr_IsValid (TopNodeElemItr* itr){
    return (itr->current_index < itr->numAdjElem);
}

void topNodeElemItr_Next (TopNodeElemItr* itr){
	itr->current_index ++;
}


TopElement topNodeElemItr_GetCurr (TopNodeElemItr* itr){
	CkAssert(topNodeElemItr_IsValid(itr));
	TopElement e;
	// TODO Make this a const reference
	ElemID elem = itr->model->mesh->n2e_getElem(itr->node, itr->current_index);
	e.id = elem.getSignedId();
	e.type = elem.getUnsignedType();
	return e;
}


void topNodeElemItr_Destroy (TopNodeElemItr* itr){
	delete itr;
}


/**************************************************************************
 *     Iterator for Facets
 * 
 *  TODO : verify that we are counting the facets correctly. 
 *               Should interior facets be found twice?
 * 				  Should boundary facets be found at all?
 * 
 */

TopFacetItr* topModel_CreateFacetItr (TopModel* m){
	TopFacetItr* itr = new TopFacetItr();
	itr->model = m;
	return itr;
}

void topFacetItr_Begin(TopFacetItr* itr){
	itr->elemItr = topModel_CreateElemItr(itr->model);
	topElemItr_Begin(itr->elemItr);
	itr->whichFacet = 0;
	
	// We break the ties between two copies of a facet by looking at the two adjacent elements
	// We must ensure that the first facet we just found is actually a valid one
	// If it is not valid, then we go to the next one

	TopElement currElem = topElemItr_GetCurr(itr->elemItr);
	ElemID e = itr->model->mesh->e2e_getElem(currElem.id, itr->whichFacet, currElem.type);
	if (e < currElem) {
		// Good, this facet is valid
	} else {
		topFacetItr_Next(itr);
	}

}

bool topFacetItr_IsValid(TopFacetItr* itr){
	return topElemItr_IsValid(itr->elemItr);
}


/** Iterate to the next facet */
void topFacetItr_Next(TopFacetItr* itr){
	bool found = false;
	
	// Scan through all the faces on some elements until we get to the end, or we 
	while( !found && topElemItr_IsValid(itr->elemItr) ){
		
		itr->whichFacet++;
		if(itr->whichFacet > 3){
			topElemItr_Next(itr->elemItr);
			itr->whichFacet=0;
		}

		if( ! topElemItr_IsValid(itr->elemItr) ){
			break;
		}

		TopElement currElem = topElemItr_GetCurr(itr->elemItr);
		ElemID e = itr->model->mesh->e2e_getElem(currElem.id, itr->whichFacet, currElem.type);
		
		if (e < currElem) {
			found = true;
//			CkPrintf("e.id=%d currElem.id=%d\n", e.id, currElem.id);
		} 
		
	}
	
}



TopFacet topFacetItr_GetCurr (TopFacetItr* itr){
	TopFacet f;
	
	TopElement el = topElemItr_GetCurr(itr->elemItr);
	f.elem[0] = el;
	
	int p1 = el.id;
	int p2 =  itr->whichFacet;
	int p3 =  el.type;

	TopElement e = itr->model->mesh->e2e_getElem(p1,p2, p3);
	
	f.elem[1] = e;
	
	// TODO adapt this to work with cohesives

	// face 0 is nodes 0,1,3
	if(itr->whichFacet==0){
		f.node[0] = f.node[3] = itr->model->mesh->elem[el.type].connFor(el.id)[0];
		f.node[1] = f.node[4] = itr->model->mesh->elem[el.type].connFor(el.id)[1];
		f.node[2] = f.node[5] = itr->model->mesh->elem[el.type].connFor(el.id)[3];
	}
	// face 1 is nodes 0,2,1
	else if(itr->whichFacet==1){
		f.node[0] = f.node[3] = itr->model->mesh->elem[el.type].connFor(el.id)[0];
		f.node[1] = f.node[4] = itr->model->mesh->elem[el.type].connFor(el.id)[2];
		f.node[2] = f.node[5] = itr->model->mesh->elem[el.type].connFor(el.id)[1];
	}
	// face 2 is nodes 1,2,3
	else if(itr->whichFacet==2){
		f.node[0] = f.node[3] = itr->model->mesh->elem[el.type].connFor(el.id)[1];
		f.node[1] = f.node[4] = itr->model->mesh->elem[el.type].connFor(el.id)[2];
		f.node[2] = f.node[5] = itr->model->mesh->elem[el.type].connFor(el.id)[3];		
	}
	// face 3 is nodes 0,3,2
	else if(itr->whichFacet==3){
		f.node[0] = f.node[3] = itr->model->mesh->elem[el.type].connFor(el.id)[0];
		f.node[1] = f.node[4] = itr->model->mesh->elem[el.type].connFor(el.id)[3];
		f.node[2] = f.node[5] = itr->model->mesh->elem[el.type].connFor(el.id)[2];
	}	
	
	return f;
}


void topFacetItr_Destroy (TopFacetItr* itr){
	delete itr;
}


#endif

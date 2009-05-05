#include "ParFUM_Iterators.h"


#ifndef INLINE_ITERATORS

/**************************************************************************
 *     Iterator for nodes
 */

MeshNodeItr*  meshModel_CreateNodeItr(MeshModel* model){
    MeshNodeItr *itr = new MeshNodeItr;
    itr->model = model;
    return itr;
}

void meshNodeItr_Destroy(MeshNodeItr* itr){
    delete itr;
}

void meshNodeItr_Begin(MeshNodeItr* itr){
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

#ifdef ITERATOR_PRINT
    CkPrintf("Initializing Node Iterator to %d\n", itr->parfum_index);
#endif
}

bool meshNodeItr_IsValid(MeshNodeItr*itr){
    return itr->model->mesh->node.is_valid_any_idx(itr->parfum_index);
}

void meshNodeItr_Next(MeshNodeItr* itr){
    CkAssert(meshNodeItr_IsValid(itr));

    // advance index until we hit a valid index
    itr->parfum_index++;

    while((!itr->model->mesh->node.is_valid_any_idx(itr->parfum_index)) &&
            (itr->parfum_index < itr->model->mesh->node.size()))
        itr->parfum_index++;

    if(itr->parfum_index==itr->model->mesh->node.size()){
        itr->parfum_index = itr->model->mesh->node.size()+1000; // way past the end
    }

#ifdef ITERATOR_PRINT
    CkPrintf("Advancing Node Iterator to %d\n", itr->parfum_index);
#endif
}


MeshNode meshNodeItr_GetCurr(MeshNodeItr*itr){
    CkAssert(meshNodeItr_IsValid(itr));
    return itr->parfum_index;
}


/**************************************************************************
 *     Iterator for elements
 * 
 *  TODO: Iterate through both cohesives & bulk elements ?
 * 
 */

MeshElemItr*  meshModel_CreateElemItr(MeshModel* model){
    MeshElemItr *itr = new MeshElemItr;
    itr->model = model;
    itr->type = MESH_ELEMENT_TET4;
    return itr;
}

void meshElemItr_Destroy(MeshElemItr* itr){
    delete itr;
}

void meshElemItr_Begin(MeshElemItr* itr){
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

#ifdef ITERATOR_PRINT
    CkPrintf("Initializing elem[itr->type] Iterator to %d\n", itr->parfum_index);
#endif

}

bool meshElemItr_IsValid(MeshElemItr*itr){
    return ! itr->done;
}

void meshElemItr_Next(MeshElemItr* itr){
    CkAssert(meshElemItr_IsValid(itr));

    // advance index until we hit a valid index
    itr->parfum_index++;

    while((!itr->model->mesh->elem[MESH_ELEMENT_TET4].is_valid_any_idx(itr->parfum_index)) &&
            (itr->parfum_index < itr->model->mesh->elem[MESH_ELEMENT_TET4].size()))
        itr->parfum_index++;


    if(itr->parfum_index==itr->model->mesh->elem[MESH_ELEMENT_TET4].size()){
        itr->done = true;
    }

#ifdef ITERATOR_PRINT
    CkPrintf("Advancing Elem Iterator to %d\n", itr->parfum_index);
#endif
}


MeshElement meshElemItr_GetCurr(MeshElemItr*itr){	
    CkAssert(meshElemItr_IsValid(itr));
    MeshElement e;
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



MeshNodeElemItr* meshModel_CreateNodeElemItr (MeshModel* model, MeshNode n){
    MeshNodeElemItr *itr = new MeshNodeElemItr;

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


bool meshNodeElemItr_IsValid (MeshNodeElemItr* itr){
    return (itr->current_index < itr->numAdjElem);
}

void meshNodeElemItr_Next (MeshNodeElemItr* itr){
    itr->current_index ++;
}


MeshElement meshNodeElemItr_GetCurr (MeshNodeElemItr* itr){
    CkAssert(meshNodeElemItr_IsValid(itr));
    MeshElement e;
    // TODO Make this a const reference
    ElemID elem = itr->model->mesh->n2e_getElem(itr->node, itr->current_index);
    e.id = elem.getSignedId();
    e.type = elem.getUnsignedType();
    return e;
}


void meshNodeElemItr_Destroy (MeshNodeElemItr* itr){
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

MeshFacetItr* meshModel_CreateFacetItr (MeshModel* m){
    MeshFacetItr* itr = new MeshFacetItr();
    itr->model = m;
    return itr;
}

void meshFacetItr_Begin(MeshFacetItr* itr){
    itr->elemItr = meshModel_CreateElemItr(itr->model);
    meshElemItr_Begin(itr->elemItr);
    itr->whichFacet = 0;

    // We break the ties between two copies of a facet by looking at the two adjacent elements
    // We must ensure that the first facet we just found is actually a valid one
    // If it is not valid, then we go to the next one

    MeshElement currElem = meshElemItr_GetCurr(itr->elemItr);
    ElemID e = itr->model->mesh->e2e_getElem(currElem.id, itr->whichFacet, currElem.type);
    if (e < currElem) {
        // Good, this facet is valid
    } else {
        meshFacetItr_Next(itr);
    }

}

bool meshFacetItr_IsValid(MeshFacetItr* itr){
    return meshElemItr_IsValid(itr->elemItr);
}


/** Iterate to the next facet */
void meshFacetItr_Next(MeshFacetItr* itr){
    bool found = false;

    // Scan through all the faces on some elements until we get to the end, or we 
    while( !found && meshElemItr_IsValid(itr->elemItr) ){

        itr->whichFacet++;
        if(itr->whichFacet > 3){
            meshElemItr_Next(itr->elemItr);
            itr->whichFacet=0;
        }

        if( ! meshElemItr_IsValid(itr->elemItr) ){
            break;
        }

        MeshElement currElem = meshElemItr_GetCurr(itr->elemItr);
        ElemID e = itr->model->mesh->e2e_getElem(currElem.id, itr->whichFacet, currElem.type);

        if (e < currElem) {
            found = true;
            //			CkPrintf("e.id=%d currElem.id=%d\n", e.id, currElem.id);
        } 

    }

}



MeshFacet meshFacetItr_GetCurr (MeshFacetItr* itr){
    MeshFacet f;

    MeshElement el = meshElemItr_GetCurr(itr->elemItr);
    f.elem[0] = el;

    int p1 = el.id;
    int p2 =  itr->whichFacet;
    int p3 =  el.type;

    MeshElement e = itr->model->mesh->e2e_getElem(p1,p2, p3);

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


void meshFacetItr_Destroy (MeshFacetItr* itr){
    delete itr;
}


#endif


/**
   @file
   @brief A ParFUM "Tops" compatibility layer API Definition

   @author Isaac Dooley

   ParFUM-TOPS provides a Tops-like API for ParFUM.


   @note ::NodeAtt and ::ElemAtt are just replaced with void* for this implementation.

*/

#ifndef __PARFUM_TOPS___H
#define __PARFUM_TOPS___H

#include <ParFUM.h>
#include <ParFUM_internals.h>

#include "ParFUM_TOPS_Types.h"
#include "ParFUM_TOPS_CUDA.h"


/** The attributes used within the PTOPS layer */
#define ATT_ELEM_ID (FEM_DATA+0)
#define ATT_ELEM_N2E_CONN (FEM_DATA+1)
#define ATT_ELEM_DATA (FEM_DATA+2)

#define ATT_NODE_ID (FEM_DATA+0)
#define ATT_NODE_COORD (FEM_DATA+1)
#define ATT_NODE_DATA (FEM_DATA+2)


/** A tops model is roughly equivalent to a ParFUM FEM_Mesh object */
class TopModel{
 public:
    FEM_Mesh *mesh;
    void *mAtt;
    AllocTable2d<unsigned char> *ElemData_T;
    AllocTable2d<unsigned char> *GhostElemData_T;
    AllocTable2d<unsigned char> *NodeData_T;
    AllocTable2d<unsigned char> *GhostNodeData_T;
    AllocTable2d<int> *ElemConn_T;
    AllocTable2d<FP_TYPE_LOW> *coord_T;
    AllocTable2d<int> *node_id_T;
    AllocTable2d<int> *elem_id_T;
    AllocTable2d<int> *n2eConn_T;

    // table mapping global ids to local indices
    CkHashtableT<CkHashtableAdaptorT<int>, int>* nodeIDHash;
    CkHashtableT<CkHashtableAdaptorT<int>, int>* elemIDHash;

    unsigned node_attr_size;
    unsigned elem_attr_size;
    unsigned model_attr_size;

    /** number of local elements */
    unsigned num_local_elem;
    /** number of local nodes */
    unsigned num_local_node;
    /** type of device to run on */
    TopDevice target_device;

#ifdef CUDA
    TopModelDevice device_model;
#endif
    TopModel(){
      nodeIDHash = NULL;
      elemIDHash = NULL;
    }
};


/** Node Iterator */
class TopNodeItr{
public:
	/** The signed index used to refer to a ParFUM Element. Non-negatives are ghosts*/
	int parfum_index;
	/** The associated model */
	TopModel *model;
};

/** Element Iterator */
class TopElemItr{
public:
	int parfum_index;
	int type;
	TopModel *model;
	bool done;
};

/** Node->Element adjacency Iterator */
class TopNodeElemItr{
public:
	/** which elem do we next supply  */
	int current_index;
	/** How many elements are adjacent to the node */
	int numAdjElem;
	
	TopModel *model;
	
	/** The node we are iterating around */
	long node;
};


/** Facet Iterator 
 * 
 * 
 * @note implemented by iterating over all elements, and 
 * in turn iterating over all the faces of each element, 
 * emmitting facets if the current element has id < the 
 * adjacent element
 * 
 * */
class TopFacetItr{
public:
	TopModel *model;
	TopElemItr *elemItr;
	int whichFacet; // Will go from 0 to 3
};

/**
 * Return the size of the FP Type the PTops
 * library was compiled with, in bytes.
 */
int tops_lib_FP_Type_Size();



/**
 * Select the device kernels should be run on.
 */
void top_set_device(TopModel* m, TopDevice d);

/**
 * Return the device kernels should be run on.
 */
TopDevice top_target_device(TopModel* m);

/**
Create and access a Tops model. Only call from Init
Currently only one model can be created. To extend, each model must just reference a different FEM_Mesh object
*/
//TopModel* topModel_Create_Init(int ElemAtt_size, int NodeAtt_size, int ModelAtt_size);
TopModel* topModel_Create_Init();


/** Create and access a Tops model. Only call from Driver */
TopModel* topModel_Create_Driver(TopDevice target_device, int elem_attr_sz, int node_attr_sz, int model_attr_sz, void* mAtt);

/** Cleanup a model. Currently does nothing */
void topModel_Destroy(TopModel* m);

/** Insert a node */
TopNode topModel_InsertNode(TopModel*, float x, float y, float z);
TopNode topModel_InsertNode(TopModel*, double x, double y, double z);

void topModel_SuggestInitialSize(TopModel* m, unsigned numNodes, unsigned numElements);


/** Set id of a node */
void topNode_SetId(TopModel*, TopNode, TopID id);

/** Set attribute of a node */
void topNode_SetAttrib(TopModel*, TopNode, void*);

/** Insert an element */
TopElement topModel_InsertElem(TopModel*, TopElemType, TopNode*);

/** Set id of an element */
void topElement_SetId(TopModel*, TopElement, TopID id);

/** Get id of an element */
int topElement_GetId (TopModel* m, TopElement e); 

/** Set attribute of an element */
void topElement_SetAttrib(TopModel*, TopElement, void*);

/** Get node via id */
TopNode topModel_GetNodeAtId(TopModel*,TopID);

/** Get nodal attribute */
void* topNode_GetAttrib(TopModel*, TopNode);

/** Get element attribute */
void* topElement_GetAttrib(TopModel*, TopElement);

/** Get node via id */
TopNode topElement_GetNode(TopModel*,TopElement,int idx);

/** Get element via id */
//#define INLINE_GETELEMATID
#ifdef INLINE_GETELEMATID
inline TopElement topModel_GetElemAtId(TopModel*m,TopID id)
{
  TopElement e;
  e.id = m->elemIDHash->get(id)-1;
  e.type = TOP_ELEMENT_TET4;
  
  if (e.id != -1) return e;
  
  AllocTable2d<int>* ghostElem_id_T = &((FEM_DataAttribute*)m->mesh->
					elem[TOP_ELEMENT_TET4].getGhost()->lookup(ATT_ELEM_ID,""))->getInt();
  
  if(ghostElem_id_T  != NULL) {
    for(int i=0; i<ghostElem_id_T->size(); ++i) {
      if((*ghostElem_id_T)(i,0)==id){
	e.id = FEM_To_ghost_index(i);
	e.type = TOP_ELEMENT_TET4;
	return e;
      }
    }
  }
  
    e.id = -1;
    e.type = TOP_ELEMENT_TET4;

    return e;
}

#else 
TopElement topModel_GetElemAtId(TopModel*,TopID);
#endif








int topNode_GetId(TopModel* m, TopNode n);

int topModel_GetNNodes(TopModel *model);

int topElement_GetNNodes(TopModel* model, TopElement elem);
bool topElement_IsCohesive(TopModel* m, TopElement e);

void topNode_GetPosition(TopModel*model, TopNode node,float*x,float*y,float*z);
void topNode_GetPosition(TopModel*model, TopNode node,double*x,double*y,double*z);


void top_retrieve_elem_data(TopModel* m);
void top_retrieve_node_data(TopModel* m);
void top_put_elem_data(TopModel* m);
void top_put_node_data(TopModel* m);

void top_retrieve_data(TopModel* m);
void top_put_data(TopModel* m);

//==============================================================
//   New functions that have been implemented



int topFacet_GetNNodes (TopModel* m, TopFacet f);
TopNode topFacet_GetNode (TopModel* m, TopFacet f, int i);
TopElement topFacet_GetElem (TopModel* m, TopFacet f, int i);

bool topElement_IsValid(TopModel* m, TopElement e);

bool topVertex_IsBoundary (TopModel* m, TopVertex v);

TopVertex topNode_GetVertex (TopModel* m, TopNode n);

TopElement topModel_InsertCohesiveAtFacet (TopModel* m, int ElemType, TopFacet f);


// TODO: setup a correct boundary condition after loading the mesh
// TODO: fix everything to work with multiple element types


bool haveConfigurableCPUGPUMap();
bool isPartitionCPU(int partition);
bool isPartitionGPU(int partition);
int configurableCPUGPUMapNumNodes();



#ifndef INLINE_ITERATORS


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
 
 /** Get total number of elements */
 int topModel_GetNElem (TopModel* m);
 
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
 
 /** Perform sanity check on iterators. This checks to make sure that the count of the itereated elements and nodes matches that returned by ParFUM's countValid() */
 void topModel_TestIterators(TopModel*m);
 
 TopNodeElemItr* topModel_CreateNodeElemItr (TopModel* m, TopNode n);
 bool topNodeElemItr_IsValid (TopNodeElemItr* neitr);
 void topNodeElemItr_Next (TopNodeElemItr* neitr);
 TopElement topNodeElemItr_GetCurr (TopNodeElemItr* neitr);
 void topNodeElemItr_Destroy (TopNodeElemItr* neitr);

 TopFacetItr* topModel_CreateFacetItr (TopModel* m);
 void topFacetItr_Begin(TopFacetItr* itr);
 bool topFacetItr_IsValid(TopFacetItr* itr);
 void topFacetItr_Next(TopFacetItr* itr);
 TopFacet topFacetItr_GetCurr (TopFacetItr* itr);
 void topFacetItr_Destroy (TopFacetItr* itr);

#else 

/**************************************************************************
 **************************************************************************
 **************************************************************************
 *   Inlined versions of iterators
 */


/**************************************************************************
 *     Iterator for nodes
 */

inline TopNodeItr*  topModel_CreateNodeItr(TopModel* model){
    TopNodeItr *itr = new TopNodeItr;
    itr->model = model;
    return itr;
}

inline void topNodeItr_Destroy(TopNodeItr* itr){
    delete itr;
}

inline void topNodeItr_Begin(TopNodeItr* itr){
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

inline bool topNodeItr_IsValid(TopNodeItr*itr){
  	return itr->model->mesh->node.is_valid_any_idx(itr->parfum_index);
}

inline void topNodeItr_Next(TopNodeItr* itr){
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


inline TopNode topNodeItr_GetCurr(TopNodeItr*itr){
	CkAssert(topNodeItr_IsValid(itr));
	return itr->parfum_index;
}


/**************************************************************************
 *     Iterator for elements
 * 
 *  TODO: Iterate through both cohesives & bulk elements ?
 * 
 */

inline TopElemItr*  topModel_CreateElemItr(TopModel* model){
    TopElemItr *itr = new TopElemItr;
    itr->model = model;
    itr->type = TOP_ELEMENT_TET4;
    return itr;
}

inline void topElemItr_Destroy(TopElemItr* itr){
    delete itr;
}

inline void topElemItr_Begin(TopElemItr* itr){
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

inline bool topElemItr_IsValid(TopElemItr*itr){
  return ! itr->done;
}

inline void topElemItr_Next(TopElemItr* itr){
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


inline TopElement topElemItr_GetCurr(TopElemItr*itr){	
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

inline TopNodeElemItr* topModel_CreateNodeElemItr (TopModel* model, TopNode n){
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


inline bool topNodeElemItr_IsValid (TopNodeElemItr* itr){
    return (itr->current_index < itr->numAdjElem);
}

inline void topNodeElemItr_Next (TopNodeElemItr* itr){
	itr->current_index ++;
}


inline TopElement topNodeElemItr_GetCurr (TopNodeElemItr* itr){
	CkAssert(topNodeElemItr_IsValid(itr));
	TopElement e;
	// TODO Make this a const reference
	ElemID elem = itr->model->mesh->n2e_getElem(itr->node, itr->current_index);
	e.id = elem.getSignedId();
	e.type = elem.getUnsignedType();
	return e;
}


inline void topNodeElemItr_Destroy (TopNodeElemItr* itr){
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

inline TopFacetItr* topModel_CreateFacetItr (TopModel* m){
	TopFacetItr* itr = new TopFacetItr();
	itr->model = m;
	return itr;
}


/** Iterate to the next facet */
inline void topFacetItr_Next(TopFacetItr* itr){
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


inline void topFacetItr_Begin(TopFacetItr* itr){
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

inline bool topFacetItr_IsValid(TopFacetItr* itr){
	return topElemItr_IsValid(itr->elemItr);
}





inline TopFacet topFacetItr_GetCurr (TopFacetItr* itr){
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


inline void topFacetItr_Destroy (TopFacetItr* itr){
	delete itr;
}

// End of Inline Iterator Section
#endif


#endif

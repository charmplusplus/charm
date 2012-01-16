/**
  @file
  @brief A ParFUM Iterators compatibility layer API

  @author Isaac Dooley
  @author Aaron Becker

  ParFUM-Iterators provides iterators for ParFUM meshes that work on
  a variety of platforms.

  @note ::NodeAtt and ::ElemAtt are just replaced with void* for this implementation.
  */

#ifndef PARFUM_ITERATORS_H
#define PARFUM_ITERATORS_H

#include <ParFUM.h>
#include <ParFUM_internals.h>

#include "ParFUM_Iterators_Types.h"
#include "ParFUM_Iterators_CUDA.h"


/** Wrappers for ParFUM attributes */
#define ATT_ELEM_ID (FEM_DATA+0)
#define ATT_ELEM_N2E_CONN (FEM_DATA+1)
#define ATT_ELEM_DATA (FEM_DATA+2)

#define ATT_NODE_ID (FEM_DATA+0)
#define ATT_NODE_COORD (FEM_DATA+1)
#define ATT_NODE_DATA (FEM_DATA+2)


/** A mesh model is roughly equivalent to a ParFUM FEM_Mesh object */
class MeshModel{
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
        MeshDevice target_device;

#ifdef CUDA
	bool allocatedForCUDADevice;
        MeshModelDevice device_model;
#endif
        MeshModel(){
            nodeIDHash = NULL;
            elemIDHash = NULL;
#ifdef CUDA
	    allocatedForCUDADevice = false;
#endif
        }


	void print(){
	  CkPrintf("MeshModel::print() on pe %d\n", CkMyPe());
	  CkPrintf("mesh=%p\n", mesh);
	  CkPrintf("mAtt=%p\n", mAtt);
	  CkPrintf("ElemData_T = %p\n", ElemData_T );
	  CkPrintf("GhostElemData_T = %p\n", GhostElemData_T);
	  CkPrintf("NodeData_T = %p\n", NodeData_T);
          CkPrintf("GhostNodeData_T = %p\n", GhostNodeData_T);
          CkPrintf("ElemConn_T = %p\n", ElemConn_T);
          CkPrintf("coord_T = %p\n", coord_T);
          CkPrintf("node_id_T = %p\n", node_id_T);
          CkPrintf("elem_id_T = %p\n", elem_id_T);
          CkPrintf("n2eConn_T = %p\n", n2eConn_T);

          CkPrintf("nodeIDHash = %p\n", nodeIDHash);
	  CkPrintf("elemIDHash = %p\n", elemIDHash);

	  CkPrintf("node_attr_size = %d\n", node_attr_size);
	  CkPrintf("elem_attr_size = %d\n", elem_attr_size);
          CkPrintf("model_attr_size = %d\n", model_attr_size);    
          CkPrintf("num_local_elem = %d\n", num_local_elem);
          CkPrintf("num_local_node = %d\n", num_local_node);
          CkPrintf("target_device = %d\n", target_device);

	}

};



// 
#ifdef CUDA
void allocateModelForCUDADevice(MeshModel* model);
void deallocateModelForCUDADevice(MeshModel* model);
#endif



/** Node Iterator */
class MeshNodeItr{
    public:
        /** The signed index used to refer to a ParFUM Element. Non-negatives are ghosts*/
        int parfum_index;
        /** The associated model */
        MeshModel *model;
};

/** Element Iterator */
class MeshElemItr{
    public:
        int parfum_index;
        int type;
        MeshModel *model;
        bool done;
};

/** Node->Element adjacency Iterator */
class MeshNodeElemItr{
    public:
        /** which elem do we next supply  */
        int current_index;
        /** How many elements are adjacent to the node */
        int numAdjElem;

        MeshModel *model;

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
class MeshFacetItr{
    public:
        MeshModel *model;
        MeshElemItr *elemItr;
        int whichFacet; // Will go from 0 to 3
};

/** 
 * Return the size of the FP Type the library was compiled with, in bytes.
 */
int lib_FP_Type_Size();



/**
 * Select the device kernels should be run on.
 */
void mesh_set_device(MeshModel* m, MeshDevice d);

/**
 * Return the device kernels should be run on.
 */
MeshDevice mesh_target_device(MeshModel* m);

/**
  Create and access a mesh model. Only call from Init
  Currently only one model can be created. To extend, each
  model must just reference a different FEM_Mesh object
  */
MeshModel* meshModel_Create_Init();


/** Create and access a mesh model. Only call from Driver */
void meshModel_Create_Driver(MeshDevice target_device,
        int elem_attr_sz, int node_attr_sz, int model_attr_sz, void* mAtt, MeshModel &model);

/** Cleanup a model. Currently does nothing */
void meshModel_Destroy(MeshModel* m);

/** Insert a node */
MeshNode meshModel_InsertNode(MeshModel*, float x, float y, float z);
MeshNode meshModel_InsertNode(MeshModel*, double x, double y, double z);

void meshModel_SuggestInitialSize(MeshModel* m, unsigned numNodes, unsigned numElements);


/** Set id of a node */
void meshNode_SetId(MeshModel*, MeshNode, EntityID id);

/** Set attribute of a node */
void meshNode_SetAttrib(MeshModel*, MeshNode, void*);

/** Insert an element */
MeshElement meshModel_InsertElem(MeshModel*, MeshElementType, MeshNode*);

/** Set id of an element */
void meshElement_SetId(MeshModel*, MeshElement, EntityID id);

/** Get id of an element */
int meshElement_GetId (MeshModel* m, MeshElement e); 

/** Set attribute of an element */
void meshElement_SetAttrib(MeshModel*, MeshElement, void*);

/** Get node via id */
MeshNode meshModel_GetNodeAtId(MeshModel*,EntityID);

/** Get nodal attribute */
void* meshNode_GetAttrib(MeshModel*, MeshNode);

/** Get element attribute */
void* meshElement_GetAttrib(MeshModel*, MeshElement);

/** Get node via id */
MeshNode meshElement_GetNode(MeshModel*,MeshElement,int idx);

/** Get element via id */
//#define INLINE_GETELEMATID
#ifdef INLINE_GETELEMATID
inline MeshElement meshModel_GetElemAtId(MeshModel*m,EntityID id)
{
    MeshElement e;
    e.id = m->elemIDHash->get(id)-1;
    e.type = MESH_ELEMENT_TET4;

    if (e.id != -1) return e;

    AllocTable2d<int>* ghostElem_id_T = &((FEM_DataAttribute*)m->mesh->
            elem[MESH_ELEMENT_TET4].getGhost()->lookup(ATT_ELEM_ID,""))->getInt();

    if(ghostElem_id_T  != NULL) {
        for(int i=0; i<ghostElem_id_T->size(); ++i) {
            if((*ghostElem_id_T)(i,0)==id){
                e.id = FEM_To_ghost_index(i);
                e.type = MESH_ELEMENT_TET4;
                return e;
            }
        }
    }

    e.id = -1;
    e.type = MESH_ELEMENT_TET4;

    return e;
}

#else 
MeshElement meshModel_GetElemAtId(MeshModel*,EntityID);
#endif


int meshNode_GetId(MeshModel* m, MeshNode n);

int meshModel_GetNNodes(MeshModel *model);

int meshElement_GetNNodes(MeshModel* model, MeshElement elem);
bool meshElement_IsCohesive(MeshModel* m, MeshElement e);

void meshNode_GetPosition(MeshModel*model, MeshNode node,float*x,float*y,float*z);
void meshNode_GetPosition(MeshModel*model, MeshNode node,double*x,double*y,double*z);


void mesh_retrieve_elem_data(MeshModel* m);
void mesh_retrieve_node_data(MeshModel* m);
void mesh_put_elem_data(MeshModel* m);
void mesh_put_node_data(MeshModel* m);

void mesh_retrieve_data(MeshModel* m);
void mesh_put_data(MeshModel* m);


int meshFacet_GetNNodes (MeshModel* m, MeshFacet f);
MeshNode meshFacet_GetNode (MeshModel* m, MeshFacet f, int i);
MeshElement meshFacet_GetElem (MeshModel* m, MeshFacet f, int i);

bool meshElement_IsValid(MeshModel* m, MeshElement e);

bool meshVertex_IsBoundary (MeshModel* m, MeshVertex v);

MeshVertex meshNode_GetVertex (MeshModel* m, MeshNode n);

MeshElement meshModel_InsertCohesiveAtFacet (MeshModel* m, int ElemType, MeshFacet f);


// TODO: setup a correct boundary condition after loading the mesh
// TODO: fix everything to work with multiple element types


bool haveConfigurableCPUGPUMap();
bool isPartitionCPU(int partition);
bool isPartitionGPU(int partition);
int configurableCPUGPUMapNumNodes();



#ifndef INLINE_ITERATORS


/** Create Iterator for nodes */
MeshNodeItr*  meshModel_CreateNodeItr(MeshModel*);

/** Destroy Iterator */
void meshNodeItr_Destroy(MeshNodeItr*);

/** Initialize Iterator */
void meshNodeItr_Begin(MeshNodeItr*);

/** Determine if Iterator is valid or if it has iterated past last Node */
bool meshNodeItr_IsValid(MeshNodeItr*);

/** Increment iterator */
void meshNodeItr_Next(MeshNodeItr*);

/** Get MeshNode associated with the iterator */
MeshNode meshNodeItr_GetCurr(MeshNodeItr*);

/** Get total number of elements */
int meshModel_GetNElem (MeshModel* m);

/** Create Iterator for elements */
MeshElemItr*  meshModel_CreateElemItr(MeshModel*);

/** Destroy Iterator */
void meshElemItr_Destroy(MeshElemItr*);

/** Initialize Iterator */
void meshElemItr_Begin(MeshElemItr*);

/** Determine if Iterator is valid or if it has iterated past last Element */
bool meshElemItr_IsValid(MeshElemItr*);

/** Increment iterator */
void meshElemItr_Next(MeshElemItr*);

/** Get MeshElement associated with the iterator */
MeshElement meshElemItr_GetCurr(MeshElemItr*);

/** Perform sanity check on iterators. This checks to make sure that the count of the itereated elements and nodes matches that returned by ParFUM's countValid() */
void meshModel_TestIterators(MeshModel*m);

MeshNodeElemItr* meshModel_CreateNodeElemItr (MeshModel* m, MeshNode n);
bool meshNodeElemItr_IsValid (MeshNodeElemItr* neitr);
void meshNodeElemItr_Next (MeshNodeElemItr* neitr);
MeshElement meshNodeElemItr_GetCurr (MeshNodeElemItr* neitr);
void meshNodeElemItr_Destroy (MeshNodeElemItr* neitr);

MeshFacetItr* meshModel_CreateFacetItr (MeshModel* m);
void meshFacetItr_Begin(MeshFacetItr* itr);
bool meshFacetItr_IsValid(MeshFacetItr* itr);
void meshFacetItr_Next(MeshFacetItr* itr);
MeshFacet meshFacetItr_GetCurr (MeshFacetItr* itr);
void meshFacetItr_Destroy (MeshFacetItr* itr);

#else 

/**************************************************************************
 **************************************************************************
 *   Inlined versions of iterators
 */


/**************************************************************************
 *     Iterator for nodes
 */

inline MeshNodeItr*  meshModel_CreateNodeItr(MeshModel* model){
    MeshNodeItr *itr = new MeshNodeItr;
    itr->model = model;
    return itr;
}

inline void meshNodeItr_Destroy(MeshNodeItr* itr){
    delete itr;
}

inline void meshNodeItr_Begin(MeshNodeItr* itr){
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

inline bool meshNodeItr_IsValid(MeshNodeItr*itr){
    return itr->model->mesh->node.is_valid_any_idx(itr->parfum_index);
}

inline void meshNodeItr_Next(MeshNodeItr* itr){
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


inline MeshNode meshNodeItr_GetCurr(MeshNodeItr*itr){
    CkAssert(meshNodeItr_IsValid(itr));
    return itr->parfum_index;
}


/**************************************************************************
 *     Iterator for elements
 * 
 *  TODO: Iterate through both cohesives & bulk elements ?
 * 
 */

inline MeshElemItr*  meshModel_CreateElemItr(MeshModel* model){
    MeshElemItr *itr = new MeshElemItr;
    itr->model = model;
    itr->type = MESH_ELEMENT_TET4;
    return itr;
}

inline void meshElemItr_Destroy(MeshElemItr* itr){
    delete itr;
}

inline void meshElemItr_Begin(MeshElemItr* itr){
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

inline bool meshElemItr_IsValid(MeshElemItr*itr){
    return ! itr->done;
}

inline void meshElemItr_Next(MeshElemItr* itr){
    CkAssert(meshElemItr_IsValid(itr));

    // advance index until we hit a valid index
    itr->parfum_index++;

    while((!itr->model->mesh->elem[MESH_ELEMENT_TET4].is_valid_any_idx(itr->parfum_index)) &&
            (itr->parfum_index < itr->model->mesh->elem[MESH_ELEMENT_TET4].size())) {
        itr->parfum_index++;
    }


    if(itr->parfum_index==itr->model->mesh->elem[MESH_ELEMENT_TET4].size()) {
        itr->done = true;
    }

#ifdef ITERATOR_PRINT
    CkPrintf("Advancing Elem Iterator to %d\n", itr->parfum_index);
#endif
}


inline MeshElement meshElemItr_GetCurr(MeshElemItr*itr){	
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

inline MeshNodeElemItr* meshModel_CreateNodeElemItr (MeshModel* model, MeshNode n){
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


inline bool meshNodeElemItr_IsValid (MeshNodeElemItr* itr){
    return (itr->current_index < itr->numAdjElem);
}

inline void meshNodeElemItr_Next (MeshNodeElemItr* itr){
    itr->current_index ++;
}


inline MeshElement meshNodeElemItr_GetCurr (MeshNodeElemItr* itr){
    CkAssert(meshNodeElemItr_IsValid(itr));
    MeshElement e;
    // TODO Make this a const reference
    ElemID elem = itr->model->mesh->n2e_getElem(itr->node, itr->current_index);
    e.id = elem.getSignedId();
    e.type = elem.getUnsignedType();
    return e;
}


inline void meshNodeElemItr_Destroy (MeshNodeElemItr* itr){
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

inline MeshFacetItr* meshModel_CreateFacetItr (MeshModel* m){
    MeshFacetItr* itr = new MeshFacetItr();
    itr->model = m;
    return itr;
}


/** Iterate to the next facet */
inline void meshFacetItr_Next(MeshFacetItr* itr){
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


inline void meshFacetItr_Begin(MeshFacetItr* itr){
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

inline bool meshFacetItr_IsValid(MeshFacetItr* itr){
    return meshElemItr_IsValid(itr->elemItr);
}

inline MeshFacet meshFacetItr_GetCurr (MeshFacetItr* itr){
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


inline void meshFacetItr_Destroy (MeshFacetItr* itr){
    delete itr;
}

// End of Inline Iterator Section
#endif




void setTableReferences(MeshModel* model, bool recomputeHash=false);


#endif

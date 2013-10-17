/**

   @file
   @brief Implementation of ParFUM-TOPS layer, except for Iterators

   @author Isaac Dooley
   @author Aaron Becker

   @todo add code to generate ghost layers
   @todo Support multiple models
   @todo Specify element types to be used via input vector in topModel_Create

*/


#include "ParFUM_TOPS.h"
#include "ParFUM.decl.h"
#include "ParFUM.h"
#include "ParFUM_internals.h"
#ifdef CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
#endif

#include <stack>
#include <sstream>
#include <iostream>


#undef DEBUG
#define DEBUG 0


int tetFaces[] = {0,1,3,  0,2,1,  1,2,3,   0,3,2};
int cohFaces[] = {0,1,2,  3,4,5};  


int tops_lib_FP_Type_Size()
{
    static const int LIB_FP_TYPE_SIZE = sizeof(FP_TYPE);
    return LIB_FP_TYPE_SIZE;
}


void top_set_device(TopModel* m, TopDevice d)
{
    m->target_device = d;
}


TopDevice top_target_device(TopModel* m)
{
    return m->target_device;
}


void fillIDHash(TopModel* model)
{

  if(model->nodeIDHash == NULL)
    model->nodeIDHash = new CkHashtableT<CkHashtableAdaptorT<int>, int>;
  
  if(model->elemIDHash == NULL)
    model->elemIDHash = new CkHashtableT<CkHashtableAdaptorT<int>, int>;
  
  for(int i=0; i<model->node_id_T->size(); ++i){
    model->nodeIDHash->put((*model->node_id_T)(i,0)) = i+1;
  }
  for(int i=0; i<model->elem_id_T->size(); ++i){
    model->elemIDHash->put((*model->elem_id_T)(i,0)) = i+1;
  }
}


// Set the pointers in the model to point to the data stored by the ParFUM framework.
// If the number of nodes or elements increases, then this function should be called
// because the attribute arrays may have been resized, after which the old pointers
// would be invalid.
void setTableReferences(TopModel* model, bool recomputeHash=false)
{
  model->ElemConn_T = &((FEM_IndexAttribute*)model->mesh->elem[TOP_ELEMENT_TET4].lookup(FEM_CONN,""))->get();
  model->elem_id_T = &((FEM_DataAttribute*)model->mesh->elem[TOP_ELEMENT_TET4].lookup(ATT_ELEM_ID,""))->getInt();
  model->n2eConn_T = &((FEM_DataAttribute*)model->mesh->elem[TOP_ELEMENT_TET4].lookup(ATT_ELEM_N2E_CONN, ""))->getInt();

  model->node_id_T = &((FEM_DataAttribute*)model->mesh->node.lookup(ATT_NODE_ID,""))->getInt();

#ifdef FP_TYPE_FLOAT
  model->coord_T = &((FEM_DataAttribute*)model->mesh->node.lookup(ATT_NODE_COORD, ""))->getFloat();
#else
  model->coord_T = &((FEM_DataAttribute*)model->mesh->node.lookup(ATT_NODE_COORD, ""))->getDouble();
#endif

   
  model->ElemData_T = &((FEM_DataAttribute*)model->mesh->elem[TOP_ELEMENT_TET4].lookup(ATT_ELEM_DATA,""))->getChar();
  FEM_Entity* ghost = model->mesh->elem[TOP_ELEMENT_TET4].getGhost();
  if (ghost)
    model->GhostElemData_T = &((FEM_DataAttribute*)ghost->lookup(ATT_ELEM_DATA,""))->getChar();
  
  model->NodeData_T = &((FEM_DataAttribute*)model->mesh->node.lookup(ATT_NODE_DATA,""))->getChar();
  ghost = model->mesh->node.getGhost();
  if (ghost)
    model->GhostNodeData_T = &((FEM_DataAttribute*)ghost->lookup(ATT_NODE_DATA,""))->getChar();
 

  if(model->nodeIDHash == NULL)
    model->nodeIDHash = new CkHashtableT<CkHashtableAdaptorT<int>, int>;
  
  if(model->elemIDHash == NULL)
    model->elemIDHash = new CkHashtableT<CkHashtableAdaptorT<int>, int>;

  if(recomputeHash){
    fillIDHash(model);
  }


}


/** Create a model  before partitioning. Given the number of nodes per element.
    
    After this call, the node data and element data CANNOT be set. They can only
    be set in the driver. If the user tries to set the attribute values, the 
    call will be ignored.

 */
TopModel* topModel_Create_Init(){
  TopModel* model = new TopModel;
  memset((void*) model, 0, sizeof(TopModel));
  
  model->nodeIDHash = new CkHashtableT<CkHashtableAdaptorT<int>, int>;
  model->elemIDHash = new CkHashtableT<CkHashtableAdaptorT<int>, int>;
  
  // This only uses a single mesh
  int which_mesh=FEM_Mesh_default_write();
  model->mesh = FEM_Mesh_lookup(which_mesh,"topModel_Create_Init");

/** @note   Here we allocate the arrays with a single
            initial node and element, which are set as
            invalid. If no initial elements were provided.
            the AllocTable2d's would not ever get allocated,
            and insertNode or insertElement would fail.
 */
  char temp_array[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


  // Allocate node id array
  FEM_Mesh_data(which_mesh,FEM_NODE,ATT_NODE_ID,temp_array, 0, 1, FEM_INT, 1);

  // Allocate node coords
#ifdef FP_TYPE_FLOAT
  FEM_Mesh_data(which_mesh,FEM_NODE,ATT_NODE_COORD,temp_array, 0, 1, FEM_FLOAT, 3);
#else
  FEM_Mesh_data(which_mesh,FEM_NODE,ATT_NODE_COORD,temp_array, 0, 1, FEM_DOUBLE, 3);
#endif

  FEM_Mesh_data(which_mesh,FEM_NODE,FEM_COORD,temp_array, 0, 1, FEM_DOUBLE, 3);  // Needed for shared node regeneration


  // Don't allocate the ATT_NODE_DATA array because it will be large



  // Allocate element connectivity
  FEM_Mesh_data(which_mesh,FEM_ELEM+TOP_ELEMENT_TET4,FEM_CONN,temp_array, 0, 1, FEM_INDEX_0, 4);
  FEM_Mesh_data(which_mesh,FEM_ELEM+TOP_ELEMENT_COH3T3,FEM_CONN,temp_array, 0, 1, FEM_INDEX_0, 6);

  // Allocate element id array
  FEM_Mesh_data(which_mesh,FEM_ELEM+TOP_ELEMENT_TET4,ATT_ELEM_ID,temp_array, 0, 1, FEM_INT, 1);
  FEM_Mesh_data(which_mesh,FEM_ELEM+TOP_ELEMENT_COH3T3,ATT_ELEM_ID,temp_array, 0, 1, FEM_INT, 1);


  // Don't allocate the ATT_ELEM_DATA array because it will be large

  
  FEM_Mesh_allocate_valid_attr(which_mesh, FEM_NODE);
  FEM_Mesh_allocate_valid_attr(which_mesh, FEM_ELEM+TOP_ELEMENT_TET4);
  FEM_Mesh_allocate_valid_attr(which_mesh, FEM_ELEM+TOP_ELEMENT_COH3T3);

  FEM_set_entity_invalid(which_mesh, FEM_NODE, 0);
  FEM_set_entity_invalid(which_mesh, FEM_ELEM+TOP_ELEMENT_TET4, 0);
  FEM_set_entity_invalid(which_mesh, FEM_ELEM+TOP_ELEMENT_COH3T3, 0);

  // Setup the adjacency lists
  
  setTableReferences(model, true);
  return model;
}

/** Get the mesh for use in the driver. It will be partitioned already.

    In the driver routine, after getting the model from this function,
    the input data file should be reread to fill in the node and element 
    data values which were not done in init.

*/
TopModel* topModel_Create_Driver(TopDevice target_device, int elem_attr_sz,
        int node_attr_sz, int model_attr_sz, void *mAtt) {
  
  CkAssert(ATT_NODE_ID != FEM_COORD);
  CkAssert(ATT_NODE_DATA != FEM_COORD);
  
  int partition = FEM_My_partition();
  if(haveConfigurableCPUGPUMap()){
    if(isPartitionCPU(partition))
      CkPrintf("partition %d is on CPU\n", partition);
    else
      CkPrintf("partition %d is on GPU\n", partition);
  }


    // This only uses a single mesh, so don't create multiple TopModels of these
    CkAssert(elem_attr_sz > 0);
    CkAssert(node_attr_sz > 0);
    int which_mesh=FEM_Mesh_default_read();
    TopModel *model = new TopModel;
    memset((void*) model, 0, sizeof(TopModel));

    model->target_device = target_device;
    model->elem_attr_size = elem_attr_sz;
    model->node_attr_size = node_attr_sz;
    model->model_attr_size = model_attr_sz;

    model->mesh = FEM_Mesh_lookup(which_mesh,"topModel_Create_Driver");
    model->mAtt = mAtt;

    model->num_local_elem = model->mesh->elem[TOP_ELEMENT_TET4].size();
    model->num_local_node = model->mesh->node.size();

    // Allocate user model attributes
    FEM_Mesh_become_set(which_mesh);
    char* temp_array = (char*) malloc(model->num_local_elem * model->elem_attr_size);
    FEM_Mesh_data(which_mesh,FEM_ELEM+TOP_ELEMENT_TET4,ATT_ELEM_DATA,temp_array,0,model->num_local_elem,FEM_BYTE,model->elem_attr_size);
    free(temp_array);
 
    temp_array = (char*) malloc(model->num_local_node * model->node_attr_size);
    FEM_Mesh_data(which_mesh,FEM_NODE,ATT_NODE_DATA,temp_array,0,model->num_local_node,FEM_BYTE,model->node_attr_size);
    free(temp_array);


    const int connSize = model->mesh->elem[TOP_ELEMENT_TET4].getConn().width();
    temp_array = (char*) malloc(model->num_local_node * connSize);
    FEM_Mesh_data(which_mesh,FEM_ELEM+TOP_ELEMENT_TET4,ATT_ELEM_N2E_CONN,temp_array, 0, 1, FEM_INT, connSize);
    free(temp_array);

    setTableReferences(model, true);
    
    // Setup the adjacencies
    int nodesPerTuple = 3;
    int tuplesPerTet = 4;
    int tuplesPerCoh = 2;

    FEM_Add_elem2face_tuples(which_mesh, TOP_ELEMENT_TET4,  nodesPerTuple, tuplesPerTet, tetFaces);
    FEM_Add_elem2face_tuples(which_mesh, TOP_ELEMENT_COH3T3,  nodesPerTuple, tuplesPerCoh, cohFaces);

    model->mesh->createNodeElemAdj();
    model->mesh->createNodeNodeAdj();
    model->mesh->createElemElemAdj();
    

#if CUDA
    int* n2eTable;
    /** Create n2e connectivity array and copy to device global memory */
    FEM_Mesh_create_node_elem_adjacency(which_mesh);
    FEM_Mesh* mesh = FEM_Mesh_lookup(which_mesh, "topModel_Create_Driver");
    FEM_DataAttribute * at = (FEM_DataAttribute*) 
        model->mesh->elem[TOP_ELEMENT_TET4].lookup(ATT_ELEM_N2E_CONN,"topModel_Create_Driver");
    n2eTable = at->getInt().getData();

    FEM_IndexAttribute * iat = (FEM_IndexAttribute*) 
        model->mesh->elem[TOP_ELEMENT_TET4].lookup(FEM_CONN,"topModel_Create_Driver");
    int* connTable  = iat->get().getData();

    int* adjElements;
    int size;
    for (int i=0; i<model->num_local_node; ++i) {
        mesh->n2e_getAll(i, adjElements, size);
        for (int j=0; j<size; ++j) {
            for (int k=0; k<connSize+1; ++k) {
                if (connTable[connSize*adjElements[j]+k] == i) {
                    n2eTable[connSize*adjElements[j]+k] = j;
                    break;
                }
                if (k == connSize) {
                    CkPrintf("Element %d cannot find node %d in its conn [%d %d %d]\n",
                            adjElements[j], i, 
                            connTable[connSize*adjElements[j]+0], 
                            connTable[connSize*adjElements[j]+1], 
                            connTable[connSize*adjElements[j]+2]); 
                    CkAssert(false);
                }
            }
        }
        delete[] adjElements;
    }
#endif

    //for (int i=0; i<model->num_local_elem*4; ++i) {
    //    printf("%d ", connTable[i]);
    //    if ((i+1)%4 == 0) printf("\n");
    //}
    //printf("\n\n");
    //for (int i=0; i<model->num_local_elem*4; ++i) {
    //    printf("%d ", n2eTable[i]);
    //    if ((i+1)%4 == 0) printf("\n");
    //}
    FEM_Mesh_become_get(which_mesh);

#if CUDA
    if (model->target_device == DeviceGPU) {
        int size = model->num_local_elem * connSize *sizeof(int);
        cudaError_t err = cudaMalloc((void**)&(model->device_model.n2eConnDevice), size);
	if(err == cudaErrorMemoryAllocation){
	  	  CkPrintf("[%d] cudaMalloc FAILED with error cudaErrorMemoryAllocation model->device_model.n2eConnDevice in ParFUM_TOPS.cc size=%d: %s\n", CkMyPe(), size, cudaGetErrorString(err));
	}else if(err != cudaSuccess){
	  CkPrintf("[%d] cudaMalloc FAILED model->device_model.n2eConnDevice in ParFUM_TOPS.cc size=%d: %s\n", CkMyPe(), size, cudaGetErrorString(err));
	  CkAbort("cudaMalloc FAILED");
	}
        CkAssert(cudaMemcpy(model->device_model.n2eConnDevice,n2eTable,size, cudaMemcpyHostToDevice) == cudaSuccess);
    }
#endif



#if CUDA
    if (model->target_device == DeviceGPU) {
        /** copy number/sizes of nodes and elements to device structure */
        model->device_model.elem_attr_size = elem_attr_sz;
        model->device_model.node_attr_size = node_attr_sz;
        model->device_model.model_attr_size = model_attr_sz;
        model->device_model.num_local_node = model->num_local_node;
        model->device_model.num_local_elem = model->num_local_elem;



        /** Copy element Attribute array to device global memory */
        {
            FEM_DataAttribute * at = (FEM_DataAttribute*) model->mesh->elem[TOP_ELEMENT_TET4].lookup(ATT_ELEM_DATA,"topModel_Create_Driver");
            AllocTable2d<unsigned char> &dataTable  = at->getChar();
            unsigned char *ElemData = dataTable.getData();
            int size = dataTable.size()*dataTable.width();
            assert(size == model->num_local_elem * model->elem_attr_size);
            CkAssert(cudaMalloc((void**)&(model->device_model.ElemDataDevice), size) == cudaSuccess);
            CkAssert(cudaMemcpy(model->device_model.ElemDataDevice,ElemData,size,
                    cudaMemcpyHostToDevice) == cudaSuccess);
        }

        /** Copy node Attribute array to device global memory */
        {
            FEM_DataAttribute * at = (FEM_DataAttribute*) model->mesh->node.lookup(ATT_NODE_DATA,"topModel_Create_Driver");
            AllocTable2d<unsigned char> &dataTable  = at->getChar();
            unsigned char *NodeData = dataTable.getData();
            int size = dataTable.size()*dataTable.width();
            assert(size == model->num_local_node * model->node_attr_size);
             CkAssert(cudaMalloc((void**)&(model->device_model.NodeDataDevice), size) == cudaSuccess);
             CkAssert(cudaMemcpy(model->device_model.NodeDataDevice,NodeData,size,
                    cudaMemcpyHostToDevice) == cudaSuccess);
        }

        /** Copy elem connectivity array to device global memory */
        {
            FEM_IndexAttribute * at = (FEM_IndexAttribute*) model->mesh->elem[TOP_ELEMENT_TET4].lookup(FEM_CONN,"topModel_Create_Driver");
            AllocTable2d<int> &dataTable  = at->get();
            int *data = dataTable.getData();
            int size = dataTable.size()*dataTable.width()*sizeof(int);
            CkAssert(cudaMalloc((void**)&(model->device_model.ElemConnDevice), size) == cudaSuccess);
            CkAssert(cudaMemcpy(model->device_model.ElemConnDevice,data,size,
                    cudaMemcpyHostToDevice) == cudaSuccess);
        }

        /** Copy model Attribute to device global memory */
        {
          printf("Copying model attribute of size %d\n", model->model_attr_size);
            CkAssert(cudaMalloc((void**)&(model->device_model.mAttDevice),
                    model->model_attr_size) == cudaSuccess);
            CkAssert(cudaMemcpy(model->device_model.mAttDevice,model->mAtt,model->model_attr_size,
                    cudaMemcpyHostToDevice) == cudaSuccess);
        }
    }
#endif

    return model;
}

/** Copy node attribute array from CUDA device back to the ParFUM attribute */
void top_retrieve_node_data(TopModel* m){ 
#if CUDA
  CkAssert(cudaMemcpy(m->NodeData_T->getData(),
            m->device_model.NodeDataDevice,
            m->num_local_node * m->node_attr_size,
            cudaMemcpyDeviceToHost) == cudaSuccess);
#endif
}

/** Copy node attribute array to CUDA device from the ParFUM attribute */
void top_put_node_data(TopModel* m){
#if CUDA
  CkAssert(cudaMemcpy(m->device_model.NodeDataDevice,
	     m->NodeData_T->getData(),
	     m->num_local_node * m->node_attr_size,
	     cudaMemcpyHostToDevice) == cudaSuccess);
#endif
}


/** Copy element attribute array from CUDA device back to the ParFUM attribute */
void top_retrieve_elem_data(TopModel* m){
#if CUDA
  CkAssert(cudaMemcpy(m->ElemData_T->getData(),
            m->device_model.ElemDataDevice,
            m->num_local_elem * m->elem_attr_size,
            cudaMemcpyDeviceToHost) == cudaSuccess);
#endif
}


/** Copy elem attribute array to CUDA device from the ParFUM attribute */
void top_put_elem_data(TopModel* m) {
#if CUDA
  CkAssert(cudaMemcpy(m->device_model.ElemDataDevice,
	     m->ElemData_T->getData(),
	     m->num_local_elem * m->elem_attr_size,
	     cudaMemcpyHostToDevice) == cudaSuccess);
#endif
}


/** Copy node and elem attribute arrays to CUDA device from the ParFUM attribute */
void top_put_data(TopModel* m) {
#if CUDA
    top_put_node_data(m);
    top_put_elem_data(m);
    CkAssert(cudaMemcpy(m->device_model.mAttDevice,m->mAtt,m->model_attr_size,
            cudaMemcpyHostToDevice) == cudaSuccess);
#endif
}


/** Copy node and elem attribute arrays from CUDA device to the ParFUM attribute */
void top_retrieve_data(TopModel* m) {
#if CUDA
    top_retrieve_node_data(m);
    top_retrieve_elem_data(m);
    CkAssert(cudaMemcpy(m->mAtt,m->device_model.mAttDevice,m->model_attr_size,
            cudaMemcpyDeviceToHost) == cudaSuccess);
#endif
}


/** Cleanup a model */
void topModel_Destroy(TopModel* m){
#if CUDA
    if (m->target_device == DeviceGPU) {
        CkAssert(cudaFree(m->device_model.mAttDevice) == cudaSuccess);
        CkAssert(cudaFree(m->device_model.NodeDataDevice) == cudaSuccess);
        CkAssert(cudaFree(m->device_model.ElemDataDevice) == cudaSuccess);
    }
#endif
    delete m;
}


TopNode topModel_InsertNode(TopModel* m, double x, double y, double z){
  int newNode = FEM_add_node_local(m->mesh,false,false,false);
  setTableReferences(m);
  (*m->coord_T)(newNode,0)=x;
  (*m->coord_T)(newNode,1)=y;
  (*m->coord_T)(newNode,2)=z;
  return newNode;
}

TopNode topModel_InsertNode(TopModel* m, float x, float y, float z){
  int newNode = FEM_add_node_local(m->mesh,false,false,false);
  setTableReferences(m);
  (*m->coord_T)(newNode,0)=x;
  (*m->coord_T)(newNode,1)=y;
  (*m->coord_T)(newNode,2)=z;
  return newNode;
}


/** Set id of a node
@todo Make this work with ghosts
*/
void topNode_SetId(TopModel* m, TopNode n, TopID id){
  CkAssert(n>=0);
  (*m->node_id_T)(n,0)=id;
  m->nodeIDHash->put(id) = n+1;
}

/** Insert an element */
TopElement topModel_InsertElem(TopModel*m, TopElemType type, TopNode* nodes){
  CkAssert(type ==  TOP_ELEMENT_TET4 || type == TOP_ELEMENT_TET10); 
         
  TopElement newEl;
 
  if(type==TOP_ELEMENT_TET4){ 
          int conn[4]; 
          conn[0] = nodes[0]; 
          conn[1] = nodes[1]; 
          conn[2] = nodes[2]; 
          conn[3] = nodes[3]; 
          newEl.type = TOP_ELEMENT_TET4;
          newEl.id = FEM_add_element_local(m->mesh, conn, 4, type, 0,0); 
  } else if (type==TOP_ELEMENT_TET10){  
          int conn[10];  
          conn[0] = nodes[0];  
          conn[1] = nodes[1];  
          conn[2] = nodes[2];  
          conn[3] = nodes[3]; 
          conn[4] = nodes[4];   
          conn[5] = nodes[5];   
          conn[6] = nodes[6];   
          conn[7] = nodes[7];  
          conn[8] = nodes[8];   
          conn[9] = nodes[9]; 
          newEl.type =  TOP_ELEMENT_TET10;
          newEl.id = FEM_add_element_local(m->mesh, conn, 10, type, 0, 0);  
  } 
 
  setTableReferences(m);
        return newEl; 
}

/** Set id of an element
@todo Make this work with ghosts
*/
void topElement_SetId(TopModel* m, TopElement e, TopID id){
  CkAssert(e.id>=0);
  (*m->elem_id_T)(e.id,0)=id;
  m->elemIDHash->put(id) = e.id+1;
}


/** get the number of elements in the mesh */
int topModel_GetNElem (TopModel* m){
	const int numBulk = m->mesh->elem[TOP_ELEMENT_TET4].count_valid();
	const int numCohesive = m->mesh->elem[TOP_ELEMENT_COH3T3].count_valid();
	std::cout << " numBulk = " << numBulk << " numCohesive " << numCohesive << std::endl;
	return numBulk + numCohesive;
}



/**
	@brief Set attribute of a node

	The attribute passed in must be a contiguous data structure with size equal to the value node_attr_sz passed into topModel_Create_Driver() and topModel_Create_Init()

	The supplied attribute will be copied into the ParFUM attribute array "ATT_NODE_DATA. Then ParFUM will own this data. The function topNode_GetAttrib() will return a pointer to the copy owned by ParFUM. If a single material parameter attribute is used for multiple nodes, each node will get a separate copy of the array. Any subsequent modifications to the data will only be reflected at a single node.

	The user is responsible for deallocating parameter d passed into this function.

*/
void topNode_SetAttrib(TopModel* m, TopNode n, void* d)
{
  if(m->NodeData_T == NULL){
    CkPrintf("Ignoring call to topNode_SetAttrib\n");
    return;
  } else {
    unsigned char* data;
    if (n < 0) {
      assert(m->GhostNodeData_T);
      data = m->GhostNodeData_T->getData();
      n = FEM_From_ghost_index(n);
    } else {
      assert(m->NodeData_T);
      data = m->NodeData_T->getData();
    }
    memcpy(data + n*m->node_attr_size, d, m->node_attr_size);
  }
}

/** @brief Set attribute of an element
See topNode_SetAttrib() for description
*/
void topElement_SetAttrib(TopModel* m, TopElement e, void* d){
   if(m->ElemData_T == NULL){
     CkPrintf("Ignoring call to topElement_SetAttrib\n");
     return;
   } else {
     unsigned char *data;
     if (e.id < 0) {
       data = m->GhostElemData_T->getData();
       e.id = FEM_From_ghost_index(e.id);
     } else {
       data = m->ElemData_T->getData();
     }
     memcpy(data + e.id*m->elem_attr_size, d, m->elem_attr_size);
   }
}

/** @brief Get elem attribute
See topNode_SetAttrib() for description
*/
void* topElement_GetAttrib(TopModel* m, TopElement e)
{
    if(! m->mesh->elem[e.type].is_valid_any_idx(e.id))
        return NULL;
    unsigned char *data;
    if (FEM_Is_ghost_index(e.id)) {
       data = m->GhostElemData_T->getData();
       e.id = FEM_From_ghost_index(e.id);
    } else {
       data = m->ElemData_T->getData();
    }
    return (data + e.id*m->elem_attr_size);
}

/** @brief Get nodal attribute
See topNode_SetAttrib() for description
*/
void* topNode_GetAttrib(TopModel* m, TopNode n)
{
    if(!m->mesh->node.is_valid_any_idx(n))
        return NULL;

    unsigned char* data;
    if (FEM_Is_ghost_index(n)) {
        data = m->GhostNodeData_T->getData();
        n = FEM_From_ghost_index(n);
    } else {
        data = m->NodeData_T->getData();
    }
    return (data + n*m->node_attr_size);
}



/**
	Get node via id
*/
TopNode topModel_GetNodeAtId(TopModel* m, TopID id)
{
    int hashnode = m->nodeIDHash->get(id)-1;
    if (hashnode != -1) return hashnode;

    AllocTable2d<int>* ghostNode_id_T = &((FEM_DataAttribute*)m->mesh->
            node.getGhost()->lookup(ATT_NODE_ID,""))->getInt();
    if(ghostNode_id_T != NULL){
      for(int i=0; i<ghostNode_id_T->size(); ++i) {
        if((*ghostNode_id_T)(i,0)==id){
	  return FEM_To_ghost_index(i);
        }
      }
    }
    return -1;
}


/**
	Get elem via id
	Note: this will currently only work with TET4 elements
 */
#ifndef INLINE_GETELEMATID
TopElement topModel_GetElemAtId(TopModel*m,TopID id)
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
#endif

TopNode topElement_GetNode(TopModel* m,TopElement e,int idx){
    int node = -1;
    if (e.id < 0) {
        CkAssert(m->mesh->elem[e.type].getGhost());
        const AllocTable2d<int> &conn = ((FEM_Elem*)m->mesh->elem[e.type].getGhost())->getConn();
        CkAssert(idx>=0 && idx<conn.width());
        node = conn(FEM_From_ghost_index(e.id),idx);
    } else {
        const AllocTable2d<int> &conn = m->mesh->elem[e.type].getConn();
        CkAssert(idx>=0 && idx<conn.width());
        node = conn(e.id,idx);
    }

    return node;
}

int topNode_GetId(TopModel* m, TopNode n){
  CkAssert(n>=0);
  return (*m->node_id_T)(n,0);
}


/** @todo handle ghost nodes as appropriate */
int topModel_GetNNodes(TopModel *model){
  return model->mesh->node.count_valid();
}

/** @todo How should we handle meshes with mixed elements? */
int topElement_GetNNodes(TopModel* model, TopElement elem){
    return model->mesh->elem[elem.type].getConn().width();
}

/** @todo make sure we are in a getting mesh */
void topNode_GetPosition(TopModel*model, TopNode node,double*x,double*y,double*z){
    if (node < 0) {
        AllocTable2d<double>* table = &((FEM_DataAttribute*)model->
                mesh->node.getGhost()->lookup(ATT_NODE_COORD,""))->getDouble();
        node = FEM_From_ghost_index(node);
        *x = (*table)(node,0);
        *y = (*table)(node,1);
        *z = (*table)(node,2);
    } else {
        *x = (*model->coord_T)(node,0);
        *y = (*model->coord_T)(node,1);
        *z = (*model->coord_T)(node,2);
    }
}

/** @todo make sure we are in a getting mesh */
void topNode_GetPosition(TopModel*model, TopNode node,float*x,float*y,float*z){
    if (node < 0) {
        AllocTable2d<float>* table = &((FEM_DataAttribute*)model->
                mesh->node.getGhost()->lookup(ATT_NODE_COORD,""))->getFloat();
        node = FEM_From_ghost_index(node);

        *x = (*table)(node,0);
        *y = (*table)(node,1);
        *z = (*table)(node,2);
    } else {
        *x = (*model->coord_T)(node,0);
        *y = (*model->coord_T)(node,1);
        *z = (*model->coord_T)(node,2);
    }
}

void topModel_Sync(TopModel*m){
  MPI_Barrier(MPI_COMM_WORLD);


  //  CkPrintf("%d: %d local, %d ghost elements\n", FEM_My_partition(), m->mesh->elem[TOP_ELEMENT_TET4].size(),m->mesh->elem[TOP_ELEMENT_TET4].ghost->size() );
  //  CkPrintf("%d: %d local, %d ghost valid elements\n", FEM_My_partition(), m->mesh->elem[TOP_ELEMENT_TET4].count_valid(),m->mesh->elem[TOP_ELEMENT_TET4].ghost->count_valid() );

}

/** Test the node and element iterators */
void topModel_TestIterators(TopModel*m){
  CkAssert(m->mesh->elem[TOP_ELEMENT_TET4].ghost!=NULL);
  CkAssert(m->mesh->node.ghost!=NULL);

  int expected_elem_count = m->mesh->elem[TOP_ELEMENT_TET4].count_valid() + m->mesh->elem[TOP_ELEMENT_TET4].ghost->count_valid();
  int iterated_elem_count = 0;

  int expected_node_count = m->mesh->node.count_valid() + m->mesh->node.ghost->count_valid();
  int iterated_node_count = 0;

  int myId = FEM_My_partition();


  TopNodeItr* itr = topModel_CreateNodeItr(m);
  for(topNodeItr_Begin(itr);topNodeItr_IsValid(itr);topNodeItr_Next(itr)){
	iterated_node_count++;
	TopNode node = topNodeItr_GetCurr(itr);
	void* na = topNode_GetAttrib(m,node);
	CkAssert(na != NULL);
  }

  TopElemItr* e_itr = topModel_CreateElemItr(m);
  for(topElemItr_Begin(e_itr);topElemItr_IsValid(e_itr);topElemItr_Next(e_itr)){
      iterated_elem_count++;
      TopElement elem = topElemItr_GetCurr(e_itr);
      void* ea = topElement_GetAttrib(m,elem);
      CkAssert(ea != NULL);
  }

  CkAssert(iterated_node_count == expected_node_count);
  CkAssert(iterated_elem_count==expected_elem_count);
  CkPrintf("Completed Iterator Test!\n");
}



bool topElement_IsCohesive(TopModel* m, TopElement e){
	return e.type > TOP_ELEMENT_MIN_COHESIVE;
}



/** currently we only support linear tets for the bulk elements */
int topFacet_GetNNodes (TopModel* m, TopFacet f){
	return 6;
}

TopNode topFacet_GetNode (TopModel* m, TopFacet f, int i){
	return f.node[i];
}

TopElement topFacet_GetElem (TopModel* m, TopFacet f, int i){
	return f.elem[i];
}

/** I'm not quite sure the point of this function
 * TODO figure out what this is supposed to do
 */
bool topElement_IsValid(TopModel* m, TopElement e){
 return m->mesh->elem[e.type].is_valid_any_idx(e.id);
}



/** We will use the following to identify the original boundary nodes. 
 * These are those that are adjacent to a facet that is on the boundary(has one adjacent element).
 * Assume vertex=node.
 */

bool topVertex_IsBoundary (TopModel* m, TopVertex v){
	return m->mesh->node.isBoundary(v);
}


TopVertex topNode_GetVertex (TopModel* m, TopNode n){
	return n;
}


int topElement_GetId (TopModel* m, TopElement e) {
  CkAssert(e.id>=0);
  return (*m->elem_id_T)(e.id,0);
}

/** Determine if two triangles are the same, but possibly varied under rotation/mirroring */
bool areSameTriangle(int a1, int a2, int a3, int b1, int b2, int b3){
	if(a1==b1 && a2==b2 && a3==b3) 
		return true;
	if(a1==b2 && a2==b3 && a3==b1) 
		return true;
	if(a1==b3 && a2==b1 && a3==b2) 
		return true;
	if(a1==b1 && a2==b3 && a3==b2) 
		return true;
	if(a1==b2 && a2==b1 && a3==b3) 
		return true;
	if(a1==b3 && a2==b2 && a3==b1) 
		return true;
	return false;
}


TopElement topModel_InsertCohesiveAtFacet (TopModel* m, int ElemType, TopFacet f){
	TopElement newCohesiveElement;
	
	CkAssert(ElemType == TOP_ELEMENT_COH3T3);

	const TopElement firstElement = f.elem[0];
	const TopElement secondElement = f.elem[1];

	CkAssert(firstElement.type != TOP_ELEMENT_COH3T3);
	CkAssert(secondElement.type != TOP_ELEMENT_COH3T3);
		
	CkAssert(firstElement.id != -1);
	CkAssert(secondElement.id != -1);

	// Create a new element
	int newEl = m->mesh->elem[ElemType].get_next_invalid(m->mesh);
	m->mesh->elem[ElemType].set_valid(newEl, false);
	
	newCohesiveElement.id = newEl;
	newCohesiveElement.type = ElemType;
	
	
#if DEBUG
	CkPrintf("/\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\/ \n");
	CkPrintf("Inserting cohesive %d of type %d at facet %d,%d,%d\n", newEl, ElemType,  f.node[0], f.node[1], f.node[2]);
#endif
	
	int conn[6];
	conn[0] = f.node[0];
	conn[1] = f.node[1];
	conn[2] = f.node[2];
	conn[3] = f.node[0];
	conn[4] = f.node[1];
	conn[5] = f.node[2];

	/// The lists of elements that can be reached from element on one side of the facet by iterating around each of the three nodes
	std::set<TopElement> reachableFromElement1[3];
	std::set<TopNode> reachableNodeFromElement1[3];
	bool canReachSecond[3];
	

	// Examine each node to determine if the node should be split
	for(int whichNode = 0; whichNode<3; whichNode++){
#if DEBUG
		CkPrintf("--------------------------------\n");
		CkPrintf("Determining whether to split node %d\n",  f.node[whichNode]);
#endif

		canReachSecond[whichNode]=false;

		TopNode theNode = f.node[whichNode];

		// Traverse across the faces to see which elements we can get to from the first element of this facet
		std::stack<TopElement> traverseTheseElements;
		CkAssert(firstElement.type != TOP_ELEMENT_COH3T3);
		traverseTheseElements.push(firstElement);

		while(traverseTheseElements.size()>0 && ! canReachSecond[whichNode]){
			TopElement traversedToElem = traverseTheseElements.top();
			traverseTheseElements.pop();

			// We should only examine elements that we have not yet already examined
			if(reachableFromElement1[whichNode].find(traversedToElem) == reachableFromElement1[whichNode].end()){
				reachableFromElement1[whichNode].insert(traversedToElem);
#if DEBUG
				CkPrintf("Can iterate from first element %d to element %d\n", firstElement.id, traversedToElem.id);
#endif
				// keep track of which nodes the split node would be adjacent to,
				// if we split this node
				for (int elemNode=0; elemNode<4; ++elemNode) {
					int queryNode = m->mesh->e2n_getNode(traversedToElem.id, elemNode, traversedToElem.type);
					if (m->mesh->n2n_exists(theNode, queryNode) &&
							queryNode != f.node[0] &&
							queryNode != f.node[1] &&
							queryNode != f.node[2]) {
						reachableNodeFromElement1[whichNode].insert(queryNode);
					}
				}
#if DEBUG
//				CkPrintf("Examining element %s,%d\n", traversedToElem.type==TOP_ELEMENT_COH3T3?"TOP_ELEMENT_COH3T3":"TOP_ELEMENT_TET4", traversedToElem.id);
#endif
				
				// Add all elements across this elements face, if they contain whichNode
				for(int face=0;face<4;face++){

					TopElement neighbor = m->mesh->e2e_getElem(traversedToElem, face); 
					// Only traverse to neighboring bulk elements
					if(neighbor.type == TOP_ELEMENT_TET4){
#if DEBUG
						CkPrintf("element %d,%d is adjacent to bulk element %d on face %d\n", traversedToElem.type,traversedToElem.id, neighbor.id, face);
#endif
						if(topElement_IsValid(m,neighbor)) {
							bool containsTheNode = false;
							for(int i=0;i<4;i++){
								if(topElement_GetNode(m,neighbor,i) == theNode){
									containsTheNode = true;
								}
							}

							if(containsTheNode){
								// Don't traverse across the face at which we are inserting the cohesive element
								if(!areSameTriangle(f.node[0],f.node[1],f.node[2],  
										topElement_GetNode(m,traversedToElem,tetFaces[face*3+0]),
										topElement_GetNode(m,traversedToElem,tetFaces[face*3+1]),
										topElement_GetNode(m,traversedToElem,tetFaces[face*3+2]) ) ){

									// If this element is the second element adjacent to the new cohesive element, we can stop
									if(neighbor == secondElement){
										canReachSecond[whichNode] = true;
#if DEBUG
										CkPrintf("We have traversed to the other side of the facet\n");
#endif
									} else {
										// Otherwise, add this element to the set remaining to be examined
										CkAssert(neighbor.type != TOP_ELEMENT_COH3T3);
										traverseTheseElements.push(neighbor);
#if DEBUG
										//									CkPrintf("Adding element %d,%d to list\n", neighbor.type, neighbor.id);
#endif
									}
								} else {
									// ignore the element because it is not adjacent to the node we are considering splitting
								}
							}
						}
					}
				}
#if DEBUG
//				CkPrintf("So far we have traversed through %d elements(%d remaining)\n", reachableFromElement1[whichNode].size(), traverseTheseElements.size() );
#endif
			}
		}

	}

#if DEBUG
	CkPrintf("\n");
#endif
	
	// Now do the actual splitting of the nodes
	int myChunk = FEM_My_partition();
	for(int whichNode = 0; whichNode<3; whichNode++){
		if(canReachSecond[whichNode]){
#if DEBUG
			CkPrintf("Node %d doesn't need to be split\n", f.node[whichNode]);			
#endif
			// Do nothing
		}else {
#if DEBUG
			CkPrintf("Node %d needs to be split\n", f.node[whichNode]);
			CkPrintf("There are %d elements that will be reassigned to the new node\n",
					reachableFromElement1[whichNode].size());
#endif

			// Create a new node
			int newNode = m->mesh->node.get_next_invalid(m->mesh);
			m->mesh->node.set_valid(newNode);

			// copy its coordinates
			// TODO: copy its other data as well
			(*m->coord_T)(newNode,0) = (*m->coord_T)(conn[whichNode],0);
			(*m->coord_T)(newNode,1) = (*m->coord_T)(conn[whichNode],1);
			(*m->coord_T)(newNode,2) = (*m->coord_T)(conn[whichNode],2);

#if DEBUG
			CkPrintf("Splitting node %d into %d and %d\n", conn[whichNode], conn[whichNode], newNode);
#endif			
			// can we use nilesh's idxl aware stuff here?
			//FEM_add_node(m->mesh, int* adjacentNodes, int numAdjacentNodes, &myChunk, 1, 0);

			// relabel one node in the cohesive element to the new node
			conn[whichNode+3] = newNode;

			// relabel the appropriate old node in the elements in reachableFromElement1
			std::set<TopElement>::iterator elem;
			for (elem = reachableFromElement1[whichNode].begin(); elem != reachableFromElement1[whichNode].end(); ++elem) {
				m->mesh->e2n_replace(elem->id, conn[whichNode], newNode, elem->type);


#if DEBUG
				CkPrintf("replacing node %d with %d in elem %d\n", conn[whichNode], newNode, elem->id);
#endif
			}

			// fix node-node adjacencies
			std::set<TopNode>::iterator node;
			for (node = reachableNodeFromElement1[whichNode].begin(); node != reachableNodeFromElement1[whichNode].end(); ++node) {
				m->mesh->n2n_replace(*node, conn[whichNode], newNode);
#if DEBUG
				CkPrintf("node %d is now adjacent to %d instead of %d\n",
						*node, newNode, conn[whichNode]);
#endif
			}		
		}
	}

#if DEBUG
	m->mesh->e2e_printAll(firstElement);
	m->mesh->e2e_printAll(secondElement);
#endif
	
	// fix elem-elem adjacencies
	m->mesh->e2e_replace(firstElement, secondElement, newCohesiveElement);
	m->mesh->e2e_replace(secondElement, firstElement, newCohesiveElement);

#if DEBUG
	CkPrintf("elements %d and %d were adjacent, now both adjacent to cohesive %d instead\n",
			firstElement.id, secondElement.id, newEl);
	
	m->mesh->e2e_printAll(firstElement);
	m->mesh->e2e_printAll(secondElement);
	
#endif
	
	
	
	
	
	// set cohesive connectivity
	m->mesh->elem[newCohesiveElement.type].connIs(newEl,conn); 
#if DEBUG
	CkPrintf("Setting connectivity of new cohesive %d to [%d %d %d %d %d %d]\n\n",
			newEl, conn[0], conn[1], conn[2], conn[3], conn[4], conn[5]);
#endif
	return newCohesiveElement;
}




// #define DEBUG1


/// A class responsible for parsing the command line arguments for the PE
/// to extract the format string passed in with +ConfigurableRRMap
class ConfigurableCPUGPUMapLoader {
public:
  
  char *locations;
  int objs_per_block;
  int numNodes;

  /// labels for states used when parsing the ConfigurableRRMap from ARGV
  enum loadStatus{
    not_loaded,
    loaded_found,
    loaded_not_found
  };
  
  enum loadStatus state;
  
  ConfigurableCPUGPUMapLoader(){
    state = not_loaded;
    locations = NULL;
    objs_per_block = 0;
  }
  
  /// load configuration if possible, and return whether a valid configuration exists
  bool haveConfiguration() {
    if(state == not_loaded) {
#ifdef DEBUG1
      CkPrintf("[%d] loading ConfigurableCPUGPUMap configuration\n", CkMyPe());
#endif
      char **argv=CkGetArgv();
      char *configuration = NULL;
      bool found = CmiGetArgString(argv, "+ConfigurableCPUGPUMap", &configuration);
      if(!found){
#ifdef DEBUG1
	CkPrintf("Couldn't find +ConfigurableCPUGPUMap command line argument\n");
#endif
	state = loaded_not_found;
	return false;
      } else {

#ifdef DEBUG1
	CkPrintf("Found +ConfigurableCPUGPUMap command line argument in %p=\"%s\"\n", configuration, configuration);
#endif

	std::istringstream instream(configuration);
	CkAssert(instream.good());
	 

	// extract first integer
	instream >> objs_per_block;
	instream >> numNodes;
	CkAssert(instream.good());
	CkAssert(objs_per_block > 0);
	locations = new char[objs_per_block];
	for(int i=0;i<objs_per_block;i++){
	  CkAssert(instream.good());
	  instream >> locations[i];
	  //	  CkPrintf("location[%d] = '%c'\n", i, locations[i]);
	  CkAssert(locations[i] == 'G' || locations[i] == 'C');
	}
	state = loaded_found;
	return true;
      }

    } else {
#ifdef DEBUG1
      CkPrintf("[%d] ConfigurableCPUGPUMap has already been loaded\n", CkMyPe());
#endif
      return state == loaded_found;
    }      
     
  }
  
};

CkpvDeclare(ConfigurableCPUGPUMapLoader, myConfigGPUCPUMapLoader);

void _initConfigurableCPUGPUMap(){
  //  CkPrintf("Initializing CPUGPU Map!\n");
  CkpvInitialize(ConfigurableCPUGPUMapLoader, myConfigGPUCPUMapLoader);
}


/// Try to load the command line arguments for ConfigurableRRMap
bool haveConfigurableCPUGPUMap(){
  ConfigurableCPUGPUMapLoader &loader =  CkpvAccess(myConfigGPUCPUMapLoader);
  return loader.haveConfiguration();
}

int configurableCPUGPUMapNumNodes(){
  ConfigurableCPUGPUMapLoader &loader =  CkpvAccess(myConfigGPUCPUMapLoader);
  return loader.numNodes;
}


bool isPartitionCPU(int partition){
  ConfigurableCPUGPUMapLoader &loader =  CkpvAccess(myConfigGPUCPUMapLoader);
  int l = partition % loader.objs_per_block;
  return loader.locations[l] == 'C';
}

bool isPartitionGPU(int partition){ 
  return ! isPartitionCPU(partition);
}







#include "ParFUM_TOPS.def.h"

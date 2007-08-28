/**

   @file
   @brief Implementation of ParFUM-TOPS layer, except for Iterators

   @author Isaac Dooley

   @todo add code to generate ghost layers
   @todo Support multiple models

   @note FEM_DATA+0 holds the elemAttr or nodeAtt data
   @note FEM_DATA+1 holds the id
   @note FEM_DATA+2 holds nodal coordinates
   @note FEM_CONN holds the element connectivity

*/

#include "ParFUM_TOPS.h"
#include "ParFUM.decl.h"
#include "ParFUM_internals.h"
#ifdef CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
#endif

void setBasicTableReferences(TopModel* model)
{
  model->ElemConn_T = &((FEM_IndexAttribute*)model->mesh->elem[0].lookup(FEM_CONN,""))->get();
  model->node_id_T = &((FEM_DataAttribute*)model->mesh->node.lookup(FEM_DATA+0,""))->getInt();
  model->elem_id_T = &((FEM_DataAttribute*)model->mesh->elem[0].lookup(FEM_DATA+0,""))->getInt();
  model->n2eConn_T = &((FEM_DataAttribute*)model->mesh->elem[0].lookup(FEM_DATA+1,""))->getInt();

#ifdef FP_TYPE_FLOAT
  model->coord_T = &((FEM_DataAttribute*)model->mesh->node.lookup(FEM_DATA+1,""))->getFloat();
#else
  model->coord_T = &((FEM_DataAttribute*)model->mesh->node.lookup(FEM_DATA+1,""))->getDouble();
#endif
}

void setTableReferences(TopModel* model)
{
    setBasicTableReferences(model);
    model->ElemData_T = &((FEM_DataAttribute*)model->mesh->elem[0].lookup(FEM_DATA+2,""))->getChar();
    model->NodeData_T = &((FEM_DataAttribute*)model->mesh->node.lookup(FEM_DATA+2,""))->getChar();
}


TopModel* topModel_Create_Init(){
  TopModel *model = new TopModel;
  memset(model, 0, sizeof(TopModel));

  // This only uses a single mesh, so better not create multiple ones of these
  int which_mesh=FEM_Mesh_default_write();
  model->mesh = FEM_Mesh_lookup(which_mesh,"TopModel::TopModel()");


/** @note   Here we allocate the arrays with a single
            initial node and element, which are set as
            invalid. If no initial elements were provided.
            the AllocTable2d's would not ever get allocated,
            and insertNode or insertElement would fail.
 */
  char temp_array[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  // Allocate node coords
#ifdef FP_TYPE_FLOAT
  FEM_Mesh_data(which_mesh,FEM_NODE,FEM_DATA+1,temp_array, 0, 1, FEM_FLOAT, 3);
#else
  FEM_Mesh_data(which_mesh,FEM_NODE,FEM_DATA+1,temp_array, 0, 1, FEM_DOUBLE, 3);
#endif
  // Allocate element connectivity
  FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_CONN,temp_array, 0, 1, FEM_INDEX_0, 4);
  // Allocate element Id array
  FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_DATA+0,temp_array, 0, 1, FEM_INT, 1);
  // Allocate n2e connectivity
  FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_DATA+1,temp_array, 0, 1, FEM_INT, 4);
  // Allocate node Id array
  FEM_Mesh_data(which_mesh,FEM_NODE+0,FEM_DATA+0,temp_array, 0, 1, FEM_INT, 1);

  FEM_Mesh_allocate_valid_attr(which_mesh, FEM_NODE);
  FEM_Mesh_allocate_valid_attr(which_mesh, FEM_ELEM+0);

  FEM_set_entity_invalid(which_mesh, FEM_NODE, 0);
  FEM_set_entity_invalid(which_mesh, FEM_ELEM+0, 0);

  setBasicTableReferences(model);
  return model;
}

TopModel* topModel_Create_Driver(int elem_attr_sz, int node_attr_sz, int model_attr_sz, void *mAtt){

    // This only uses a single mesh, so don't create multiple TopModels of these
    CkAssert(elem_attr_sz > 0);
    CkAssert(node_attr_sz > 0);
    int which_mesh=FEM_Mesh_default_read();
    TopModel *model = new TopModel;
    memset(model, 0, sizeof(TopModel));

    model->elem_attr_size = elem_attr_sz;
    model->node_attr_size = node_attr_sz;
    model->model_attr_size = model_attr_sz;

    model->mesh = FEM_Mesh_lookup(which_mesh,"topModel_Create_Driver");
    model->mAtt = mAtt;

    model->num_local_elem = model->mesh->elem[0].size();
    model->num_local_node = model->mesh->node.size();

    FEM_Mesh_become_set(which_mesh);
    char* temp_array = (char*) malloc(model->num_local_elem * model->elem_attr_size);
    FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_DATA+2,temp_array,0,model->num_local_elem,FEM_BYTE,model->elem_attr_size);
    free(temp_array);

    //int mesh_write = FEM_Mesh_default_write();
    temp_array = (char*) malloc(model->num_local_node * model->node_attr_size);
    FEM_Mesh_data(which_mesh,FEM_NODE+0,FEM_DATA+2,temp_array,0,model->num_local_node,FEM_BYTE,model->node_attr_size);
    free(temp_array);
    FEM_Mesh_become_get(which_mesh);

    setTableReferences(model);

    /** Create n2e connectivity array and copy to device global memory */
    {
        FEM_Mesh_create_node_elem_adjacency(which_mesh);
        FEM_Mesh* mesh = FEM_Mesh_lookup(which_mesh, "topModel_Create_Driver");
        FEM_DataAttribute * at = (FEM_DataAttribute*) 
            model->mesh->elem[0].lookup(FEM_DATA+1,"topModel_Create_Driver");
        int* n2eTable  = at->getInt().getData();

        FEM_IndexAttribute * iat = (FEM_IndexAttribute*) 
            model->mesh->elem[0].lookup(FEM_CONN,"topModel_Create_Driver");
        int* connTable  = iat->get().getData();

        int* adjElements;
        int size;
        for (int i=0; i<model->num_local_node; ++i) {
            mesh->n2e_getAll(i, adjElements, size);
            for (int j=0; j<size; ++j) {
                for (int k=0; k<5; ++k) {
                    if (connTable[4*adjElements[j]+k] == i) {
                        n2eTable[4*adjElements[j]+k] = j;
                        break;
                    }
                    assert(k != 4);
                }
            }
            delete[] adjElements;
        }

        //for (int i=0; i<model->num_local_elem*4; ++i) {
        //    printf("%d ", connTable[i]);
        //    if ((i+1)%4 == 0) printf("\n");
        //}
        //printf("\n\n");
        //for (int i=0; i<model->num_local_elem*4; ++i) {
        //    printf("%d ", n2eTable[i]);
        //    if ((i+1)%4 == 0) printf("\n");
        //}

#if CUDA
        size = model->num_local_elem * 4 *sizeof(int);
        cudaMalloc((void**)&(model->device_model.n2eConnDevice), size);
        cudaMemcpy(model->device_model.n2eConnDevice,n2eTable,size,
                cudaMemcpyHostToDevice);
#endif

    }




#if CUDA
    /** copy number/sizes of nodes and elements to device structure */
    model->device_model.elem_attr_size = elem_attr_sz;
    model->device_model.node_attr_size = node_attr_sz;
    model->device_model.model_attr_size = model_attr_sz;
    model->device_model.num_local_node = model->num_local_node;
    model->device_model.num_local_elem = model->num_local_elem;



    /** Copy element Attribute array to device global memory */
    {
        FEM_DataAttribute * at = (FEM_DataAttribute*) model->mesh->elem[0].lookup(FEM_DATA+0,"topModel_Create_Driver");
        AllocTable2d<unsigned char> &dataTable  = at->getChar();
        unsigned char *ElemData = dataTable.getData();
        int size = dataTable.size()*dataTable.width();
        cudaMalloc((void**)&(model->device_model.ElemDataDevice), size);
        cudaMemcpy(model->device_model.ElemDataDevice,ElemData,size,
                cudaMemcpyHostToDevice);
    }

    /** Copy node Attribute array to device global memory */
    {
        FEM_DataAttribute * at = (FEM_DataAttribute*) model->mesh->node.lookup(FEM_DATA+0,"topModel_Create_Driver");
        AllocTable2d<unsigned char> &dataTable  = at->getChar();
        unsigned char *NodeData = dataTable.getData();
        int size = dataTable.size()*dataTable.width();
        cudaMalloc((void**)&(model->device_model.NodeDataDevice), size);
        cudaMemcpy(model->device_model.NodeDataDevice,NodeData,size,
                cudaMemcpyHostToDevice);
    }

    /** Copy elem connectivity array to device global memory */
    {
        FEM_IndexAttribute * at = (FEM_IndexAttribute*) model->mesh->elem[0].lookup(FEM_CONN,"topModel_Create_Driver");
        AllocTable2d<int> &dataTable  = at->get();
        int *data = dataTable.getData();
        int size = dataTable.size()*dataTable.width()*sizeof(int);
        cudaMalloc((void**)&(model->device_model.ElemConnDevice), size);
        cudaMemcpy(model->device_model.ElemConnDevice,data,size,
                cudaMemcpyHostToDevice);
    }



    /** Copy model Attribute to device global memory */
    {
      printf("Copying model attribute of size %d\n", model->model_attr_size);
        cudaMalloc((void**)&(model->device_model.mAttDevice),
                model->model_attr_size);
        cudaMemcpy(model->device_model.mAttDevice,mAtt,model->model_attr_size,
                cudaMemcpyHostToDevice);
    }

#endif

    return model;
}

/** Copy node attribute array from CUDA device back to the ParFUM attribute */
void top_retrieve_node_data(TopModel* m){ 
#if CUDA
  cudaMemcpy(m->NodeData_T->getData(),
            m->device_model.NodeDataDevice,
            m->num_local_node * m->node_attr_size,
            cudaMemcpyDeviceToHost);
#endif
}

/** Copy node attribute array to CUDA device from the ParFUM attribute */
void top_put_node_data(TopModel* m){
#if CUDA
  cudaMemcpy(m->device_model.NodeDataDevice,
	     m->NodeData_T->getData(),
	     m->num_local_node * m->node_attr_size,
	     cudaMemcpyHostToDevice);
#endif
}


/** Copy element attribute array from CUDA device back to the ParFUM attribute */
void top_retrieve_elem_data(TopModel* m){
#if CUDA
  cudaMemcpy(m->ElemData_T->getData(),
            m->device_model.ElemDataDevice,
            m->num_local_elem * m->elem_attr_size,
            cudaMemcpyDeviceToHost);
#endif
}



/** Cleanup a model */
void topModel_Destroy(TopModel* m){
#if CUDA
    cudaFree(m->device_model.mAttDevice);
    cudaFree(m->device_model.NodeDataDevice);
    cudaFree(m->device_model.ElemDataDevice);
#endif
    delete m;
}


TopNode topModel_InsertNode(TopModel* m, double x, double y, double z){
  int newNode = FEM_add_node_local(m->mesh,false,false,false);
  (*m->coord_T)(newNode,0)=x;
  (*m->coord_T)(newNode,1)=y;
  (*m->coord_T)(newNode,2)=z;
  return newNode;
}

TopNode topModel_InsertNode(TopModel* m, float x, float y, float z){
  int newNode = FEM_add_node_local(m->mesh,false,false,false);
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
}

/** Insert an element */
TopElement topModel_InsertElem(TopModel*m, TopElemType type, TopNode* nodes){
  CkAssert(type ==  TOP_ELEMENT_TET4 || type == TOP_ELEMENT_TET10); 
         
 int newEl; 
 
  if(type==TOP_ELEMENT_TET4){ 
          int conn[4]; 
          conn[0] = nodes[0]; 
          conn[1] = nodes[1]; 
          conn[2] = nodes[2]; 
          conn[3] = nodes[3]; 
          newEl = FEM_add_element_local(m->mesh, conn, 4, 0, 0, 0); 
  } else if (type==TOP_ELEMENT_TET4){  
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
          newEl = FEM_add_element_local(m->mesh, conn, 10, 0, 0, 0);  
  } 
 
         return newEl; 
}

/** Set id of an element
@todo Make this work with ghosts
*/
void topElement_SetId(TopModel* m, TopElement e, TopID id){
  CkAssert(e>=0);
  (*m->elem_id_T)(e,0)=id;
}



/**
	@brief Set attribute of a node

	The attribute passed in must be a contiguous data structure with size equal to the value node_attr_sz passed into topModel_Create_Driver() and topModel_Create_Init()

	The supplied attribute will be copied into the ParFUM attribute array "FEM_DATA+0". Then ParFUM will own this data. The function topNode_GetAttrib() will return a pointer to the copy owned by ParFUM. If a single material parameter attribute is used for multiple nodes, each node will get a separate copy of the array. Any subsequent modifications to the data will only be reflected at a single node.

	The user is responsible for deallocating parameter d passed into this function.

*/
void topNode_SetAttrib(TopModel* m, TopNode n, void* d){
  CkAssert(n>=0);
  unsigned char *data = m->NodeData_T->getData();
  memcpy(data + n*m->node_attr_size, d, m->node_attr_size);
}

/** @brief Set attribute of an element
See topNode_SetAttrib() for description
*/
void topElement_SetAttrib(TopModel* m, TopElement e, void* d){
  CkAssert(e>=0);
  unsigned char *data = m->ElemData_T->getData();
  memcpy(data + e*m->elem_attr_size, d, m->elem_attr_size);
}


/** @brief Get elem attribute
See topNode_SetAttrib() for description
*/
void* topElement_GetAttrib(TopModel* m, TopElement e){
  if(! m->mesh->elem[0].is_valid_any_idx(e))
	return NULL;
  unsigned char *data = m->ElemData_T->getData();
  return (data + e*m->elem_attr_size);
}

/** @brief Get nodal attribute
See topNode_SetAttrib() for description
*/
void* topNode_GetAttrib(TopModel* m, TopNode n){
  if(! m->mesh->node.is_valid_any_idx(n))
	return NULL;
  unsigned char *data = m->NodeData_T->getData();
  return (data + n*m->node_attr_size);
}



/**
	Get node via id
	@todo Re-implement this function with some hash to make it fast.
	@note In the explicit FEA example, this is just used during initialization, so speed is not too important.
	@todo Does not work with ghosts yet.
*/
TopNode topModel_GetNodeAtId(TopModel* m, TopID id){
  // lookup node via global ID
  for(int i=0;i<m->node_id_T->size();++i){
	if((*m->node_id_T)(i,0)==id){
	  return i;
	}
  }
  return -1;
}


/**
	Get elem via id
 */
TopElement topModel_GetElemAtId(TopModel*m,TopID id){
    // lookup node via global ID
    for(int i=0;i<m->elem_id_T->size();++i){
        if( m->mesh->elem[0].is_valid(i) && (*m->elem_id_T)(i,0)==id){
		  return i;
        }
    }
    return -1;
}


TopNode topElement_GetNode(TopModel* m,TopElement e,int idx){
  CkAssert(e>=0);
  const AllocTable2d<int> &conn = m->mesh->elem[0].getConn();
  CkAssert(idx>=0 && idx<conn.width());

  int node = conn(e,idx);

  return conn(e,idx);
}

int topNode_GetId(TopModel* m, TopNode n){
  CkAssert(n>=0);
  return (*m->node_id_T)(n,0);
}


/** @todo handle ghost nodes as appropriate */
int topModel_GetNNodes(TopModel *model){
  return model->mesh->node.count_valid();
}

/** @todo Fix to return the width of the conn array */
int topElement_GetNNodes(TopModel* model, TopElement elem){
  return 4;
}

/** @todo make sure we are in a getting mesh */
void topNode_GetPosition(TopModel*model, TopNode node,double*x,double*y,double*z){
  CkAssert(node>=0);
  *x = (*model->coord_T)(node,0);
  *y = (*model->coord_T)(node,1);
  *z = (*model->coord_T)(node,2);
}

/** @todo make sure we are in a getting mesh */
void topNode_GetPosition(TopModel*model, TopNode node,float*x,float*y,float*z){
  CkAssert(node>=0);
  *x = (*model->coord_T)(node,0);
  *y = (*model->coord_T)(node,1);
  *z = (*model->coord_T)(node,2);
}

void topModel_Sync(TopModel*m){
  MPI_Barrier(MPI_COMM_WORLD);


  //  CkPrintf("%d: %d local, %d ghost elements\n", FEM_My_partition(), m->mesh->elem[0].size(),m->mesh->elem[0].ghost->size() );
  //  CkPrintf("%d: %d local, %d ghost valid elements\n", FEM_My_partition(), m->mesh->elem[0].count_valid(),m->mesh->elem[0].ghost->count_valid() );

}

void topModel_TestIterators(TopModel*m){
  CkAssert(m->mesh->elem[0].ghost!=NULL);
  CkAssert(m->mesh->node.ghost!=NULL);

  int expected_elem_count = m->mesh->elem[0].count_valid() + m->mesh->elem[0].ghost->count_valid();
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

}


#include "ParFUM_TOPS.def.h"

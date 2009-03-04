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

int tops_lib_FP_Type_Size()
{
    static const int LIB_FP_TYPE_SIZE = sizeof(FP_TYPE);
    return LIB_FP_TYPE_SIZE;
}

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
    FEM_Entity* ghost = model->mesh->elem[0].getGhost();
    if (ghost)
        model->GhostElemData_T = &((FEM_DataAttribute*)ghost->lookup(FEM_DATA+2,""))->getChar();
    model->NodeData_T = &((FEM_DataAttribute*)model->mesh->node.lookup(FEM_DATA+2,""))->getChar();
    ghost = model->mesh->node.getGhost();
    if (ghost)
        model->GhostNodeData_T = &((FEM_DataAttribute*)ghost->lookup(FEM_DATA+2,""))->getChar();
}

void fillIDHash(TopModel* model)
{
    for(int i=0; i<model->node_id_T->size(); ++i){
        model->nodeIDHash->put((*model->node_id_T)(i,0)) = i+1;
    }
    for(int i=0; i<model->elem_id_T->size(); ++i){
        model->elemIDHash->put((*model->elem_id_T)(i,0)) = i+1;
    }
}

TopModel* topModel_Create_Init(int nelnode){
  TopModel* model = new TopModel;
  memset(model, 0, sizeof(TopModel));
  
  model->nodeIDHash = new CkHashtableT<CkHashtableAdaptorT<int>, int>;
  model->elemIDHash = new CkHashtableT<CkHashtableAdaptorT<int>, int>;
  
  // This only uses a single mesh, so better not create multiple ones of these
  int which_mesh=FEM_Mesh_default_write();
  model->mesh = FEM_Mesh_lookup(which_mesh,"topModel_Create_Init");


/** @note   Here we allocate the arrays with a single
            initial node and element, which are set as
            invalid. If no initial elements were provided.
            the AllocTable2d's would not ever get allocated,
            and insertNode or insertElement would fail.
 */
  char temp_array[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  // Allocate node coords
#ifdef FP_TYPE_FLOAT
  FEM_Mesh_data(which_mesh,FEM_NODE,FEM_DATA+1,temp_array, 0, 1, FEM_FLOAT, 3);
#else
  FEM_Mesh_data(which_mesh,FEM_NODE,FEM_DATA+1,temp_array, 0, 1, FEM_DOUBLE, 3);
#endif
  // Allocate element connectivity
  FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_CONN,temp_array, 0, 1, FEM_INDEX_0, nelnode);
  // Allocate element Id array
  FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_DATA+0,temp_array, 0, 1, FEM_INT, 1);
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

    // Allocate user model attributes
    FEM_Mesh_become_set(which_mesh);
    char* temp_array = (char*) malloc(model->num_local_elem * model->elem_attr_size);
    FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_DATA+2,temp_array,0,model->num_local_elem,FEM_BYTE,model->elem_attr_size);
    free(temp_array);
    temp_array = (char*) malloc(model->num_local_node * model->node_attr_size);
    FEM_Mesh_data(which_mesh,FEM_NODE+0,FEM_DATA+2,temp_array,0,model->num_local_node,FEM_BYTE,model->node_attr_size);
    free(temp_array);

    // Allocate n2e connectivity
    const int connSize = model->mesh->elem[0].getConn().width();
    temp_array = (char*) malloc(model->num_local_node * connSize);
    FEM_Mesh_data(which_mesh,FEM_ELEM+0,FEM_DATA+1,temp_array, 0, 1, FEM_INT, connSize);
    free(temp_array);

    setTableReferences(model);
    model->nodeIDHash = new CkHashtableT<CkHashtableAdaptorT<int>, int>;
    model->elemIDHash = new CkHashtableT<CkHashtableAdaptorT<int>, int>;
    fillIDHash(model);


#if CUDA
    /** Create n2e connectivity array and copy to device global memory */
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
    size = model->num_local_elem * connSize *sizeof(int);
    cudaMalloc((void**)&(model->device_model.n2eConnDevice), size);
    cudaMemcpy(model->device_model.n2eConnDevice,n2eTable,size,
            cudaMemcpyHostToDevice);
#endif



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
          newEl.type = BULK_ELEMENT;
          newEl.idx = FEM_add_element_local(m->mesh, conn, 4, 0, 0, 0); 
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
          newEl.type = BULK_ELEMENT;
          newEl.idx = FEM_add_element_local(m->mesh, conn, 10, 0, 0, 0);  
  } 
 
         return newEl; 
}

/** Set id of an element
@todo Make this work with ghosts
*/
void topElement_SetId(TopModel* m, TopElement e, TopID id){
  CkAssert(e.idx>=0);
  (*m->elem_id_T)(e.idx,0)=id;
  m->elemIDHash->put(id) = e.idx+1;
}


/** get the number of elements in the mesh */
int topModel_GetNElem (TopModel* m){
	const int numBulk = m->mesh->elem[BULK_ELEMENT].count_valid();
	const int numCohesive = m->mesh->elem[COHESIVE_ELEMENT].count_valid();
	return numBulk + numCohesive;
}



/**
	@brief Set attribute of a node

	The attribute passed in must be a contiguous data structure with size equal to the value node_attr_sz passed into topModel_Create_Driver() and topModel_Create_Init()

	The supplied attribute will be copied into the ParFUM attribute array "FEM_DATA+0". Then ParFUM will own this data. The function topNode_GetAttrib() will return a pointer to the copy owned by ParFUM. If a single material parameter attribute is used for multiple nodes, each node will get a separate copy of the array. Any subsequent modifications to the data will only be reflected at a single node.

	The user is responsible for deallocating parameter d passed into this function.

*/
void topNode_SetAttrib(TopModel* m, TopNode n, void* d)
{
    unsigned char* data;
    if (n < 0) {
        data = m->GhostNodeData_T->getData();
        n = FEM_From_ghost_index(n);
    } else {
        data = m->NodeData_T->getData();
    }
    memcpy(data + n*m->node_attr_size, d, m->node_attr_size);
}

/** @brief Set attribute of an element
See topNode_SetAttrib() for description
*/
void topElement_SetAttrib(TopModel* m, TopElement e, void* d){
  unsigned char *data;
  if (e.idx < 0) {
      data = m->GhostElemData_T->getData();
      e.idx = FEM_From_ghost_index(e.idx);
  } else {
    data = m->ElemData_T->getData();
  }
  memcpy(data + e.idx*m->elem_attr_size, d, m->elem_attr_size);
}


/** @brief Get elem attribute
See topNode_SetAttrib() for description
*/
void* topElement_GetAttrib(TopModel* m, TopElement e)
{
    if(! m->mesh->elem[0].is_valid_any_idx(e.idx))
        return NULL;
    unsigned char *data;
    if (FEM_Is_ghost_index(e.idx)) {
       data = m->GhostElemData_T->getData();
       e.idx = FEM_From_ghost_index(e.idx);
    } else {
       data = m->ElemData_T->getData();
    }
    return (data + e.idx*m->elem_attr_size);
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
            node.getGhost()->lookup(FEM_DATA+0,""))->getInt();
    for(int i=0; i<ghostNode_id_T->size(); ++i) {
        if((*ghostNode_id_T)(i,0)==id){
            return FEM_To_ghost_index(i);
        }
    }
    return -1;
}


/**
	Get elem via id
 */
TopElement topModel_GetElemAtId(TopModel*m,TopID id)
{
	TopElement e;
	e.idx = m->elemIDHash->get(id)-1;
	e.type = BULK_ELEMENT;
	
	if (e.idx != -1) return e;

    AllocTable2d<int>* ghostElem_id_T = &((FEM_DataAttribute*)m->mesh->
            elem[0].getGhost()->lookup(FEM_DATA+0,""))->getInt();
    for(int i=0; i<ghostElem_id_T->size(); ++i) {
        if((*ghostElem_id_T)(i,0)==id){
            e.idx = FEM_To_ghost_index(i);
            e.type = BULK_ELEMENT;
        	return e;
        }
    }
    
    e.idx = -1;
    e.type = BULK_ELEMENT;

    return e;
}


TopNode topElement_GetNode(TopModel* m,TopElement e,int idx){
    int node = -1;
    if (e.idx < 0) {
        CkAssert(m->mesh->elem[0].getGhost());
        const AllocTable2d<int> &conn = ((FEM_Elem*)m->mesh->elem[0].getGhost())->getConn();
        CkAssert(idx>=0 && idx<conn.width());
        node = conn(FEM_From_ghost_index(e.idx),idx);
    } else {
        const AllocTable2d<int> &conn = m->mesh->elem[0].getConn();
        CkAssert(idx>=0 && idx<conn.width());
        node = conn(e.idx,idx);
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
    return model->mesh->elem[0].getConn().width();
}

/** @todo make sure we are in a getting mesh */
void topNode_GetPosition(TopModel*model, TopNode node,double*x,double*y,double*z){
    if (node < 0) {
        AllocTable2d<double>* table = &((FEM_DataAttribute*)model->
                mesh->node.getGhost()->lookup(FEM_DATA+1,""))->getDouble();
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
                mesh->node.getGhost()->lookup(FEM_DATA+1,""))->getFloat();
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





bool topElement_IsCohesive(TopModel* m, TopElement e){
	return e.type == COHESIVE_ELEMENT;
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
 return m->mesh->elem[e.type].is_valid_any_idx(e.idx);
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
  CkAssert(e.idx>=0);
  return (*m->elem_id_T)(e.idx,0);
}




TopFacetItr* topModel_CreateFacetItr (TopModel* m){
	TopFacetItr* itr = new TopFacetItr();
	itr->model = m;
	return itr;
}

void topFacetItr_Begin(TopFacetItr* itr){
	itr->elemItr = topModel_CreateElemItr(itr->model);
	itr->whichFacet = 0;
}

bool topFacetItr_IsValid(TopFacetItr* itr){
	return topElemItr_IsValid(itr->elemItr);
}

/** Iterate to the next facet */
void topFacetItr_Next(TopFacetItr* itr){
	bool found = false;
	
	// Scan through all the faces on some elements until we get to the end, or we 
	while( !found && topElemItr_IsValid(itr->elemItr) ){
		found = true;
		
		itr->whichFacet++;
		if(itr->whichFacet > 3){
			topElemItr_Next(itr->elemItr);
			itr->whichFacet=0;
		}

		TopElement currElem = topElemItr_GetCurr(itr->elemItr);
		FEM_VarIndexAttribute::ID e = itr->model->mesh->e2e_getElem(currElem.idx, itr->whichFacet, currElem.type);

		// TODO Adapt to work with cohesives
		if (e.id < currElem.idx){
			found = true;
		}
	}
	
}

/** Determine whether an element contains the  given face */
//inline bool elementContainsNodes(TopModel *model, TopElement e, int n3, int n2, int n1){
//	const int *nodes = model->mesh->elem[e.type].connFor(e.idx);
//	const int e0 = nodes[0];
//	const int e1 = nodes[1];
//	const int e2 = nodes[2];
//	const int e3 = nodes[3];
//	
//	// We take in the three nodes in the original orientation in the other element,
//	// so n1,n2,n3 is in the orientation of this element
//	
//	// Examine face 0 of the element (in 3 rotations)
//	return	( n1==e0 && n2==e1 && n3==e3 ) ||
//			( n1==e1 && n2==e3 && n3==e0 ) ||
//			( n1==e3 && n2==e0 && n3==e1 ) ||
//	// Examine face 1 of the element (in 3 rotations)
//			( n1==e0 && n2==e2 && n3==e1 ) ||
//			( n1==e2 && n2==e1 && n3==e0 ) ||
//			( n1==e1 && n2==e0 && n3==e2 ) ||
//	// Examine face 2 of the element (in 3 rotations)
//			( n1==e1 && n2==e2 && n3==e3 ) ||
//			( n1==e2 && n2==e3 && n3==e1 ) ||
//			( n1==e3 && n2==e1 && n3==e2 ) ||
//	// Examine face 3 of the element (in 3 rotations)
//			( n1==e0 && n2==e3 && n3==e2 ) ||
//			( n1==e3 && n2==e2 && n3==e0 ) ||
//			( n1==e2 && n2==e0 && n3==e3 ) ;
//
//}


TopFacet topFacetItr_GetCurr (TopFacetItr* itr){
	TopFacet f;
	
	f.elem[0] = topElemItr_GetCurr(itr->elemItr);
	FEM_VarIndexAttribute::ID e = itr->model->mesh->e2e_getElem(f.elem[0].idx, itr->whichFacet, f.elem[0].type);
	f.elem[1].idx = e.id;
	f.elem[1].type = e.type;
	
	// TODO adapt this to work with cohesives

	// face 0 is nodes 0,1,3
	if(itr->whichFacet==0){
		f.node[0] = f.node[3] = itr->model->mesh->elem[f.elem[0].type].connFor(f.elem[0].idx)[0];
		f.node[1] = f.node[4] = itr->model->mesh->elem[f.elem[0].type].connFor(f.elem[0].idx)[1];
		f.node[2] = f.node[5] = itr->model->mesh->elem[f.elem[0].type].connFor(f.elem[0].idx)[3];
	}
	// face 1 is nodes 0,2,1
	else if(itr->whichFacet==1){
		f.node[0] = f.node[3] = itr->model->mesh->elem[f.elem[0].type].connFor(f.elem[0].idx)[0];
		f.node[1] = f.node[4] = itr->model->mesh->elem[f.elem[0].type].connFor(f.elem[0].idx)[2];
		f.node[2] = f.node[5] = itr->model->mesh->elem[f.elem[0].type].connFor(f.elem[0].idx)[1];
	}
	// face 2 is nodes 1,2,3
	else if(itr->whichFacet==2){
		f.node[0] = f.node[3] = itr->model->mesh->elem[f.elem[0].type].connFor(f.elem[0].idx)[1];
		f.node[1] = f.node[4] = itr->model->mesh->elem[f.elem[0].type].connFor(f.elem[0].idx)[2];
		f.node[2] = f.node[5] = itr->model->mesh->elem[f.elem[0].type].connFor(f.elem[0].idx)[3];		
	}
	// face 3 is nodes 0,3,2
	else if(itr->whichFacet==3){
		f.node[0] = f.node[3] = itr->model->mesh->elem[f.elem[0].type].connFor(f.elem[0].idx)[0];
		f.node[1] = f.node[4] = itr->model->mesh->elem[f.elem[0].type].connFor(f.elem[0].idx)[3];
		f.node[2] = f.node[5] = itr->model->mesh->elem[f.elem[0].type].connFor(f.elem[0].idx)[2];
	}	
	
	return f;
}


void topFacetItr_Destroy (TopFacetItr* itr){
	delete itr;
}


#include "ParFUM_TOPS.def.h"

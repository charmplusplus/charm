/*Charm++ Finite Element Framework:
C interface file
*/
#ifndef _CHARM_FEM_H
#define _CHARM_FEM_H
#include "pup_c.h"  /* for pup_er */
#include "idxlc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* datatypes: keep in sync with femf.h and idxl */
#define FEM_BYTE   IDXL_BYTE
#define FEM_INT    IDXL_INT
#define FEM_REAL   IDXL_REAL
#define FEM_FLOAT FEM_REAL /*alias*/
#define FEM_DOUBLE IDXL_DOUBLE
#define FEM_INDEX_0  IDXL_INDEX_0
#define FEM_INDEX_1  IDXL_INDEX_1 
#define FEM_VAR_INDEX (IDXL_FIRST_DATATYPE+6)

/* reduction operations: keep in sync with femf.h */
#define FEM_SUM IDXL_SUM
#define FEM_PROD IDXL_PROD
#define FEM_MAX IDXL_MAX
#define FEM_MIN IDXL_MIN

/* element types, by their number of nodes */
#define FEM_TRIANGULAR    3
#define FEM_TETRAHEDRAL   4
#define FEM_HEXAHEDRAL    8
#define FEM_QUADRILATERAL 4

/* initialization flags */
#define FEM_INIT_READ    2
#define FEM_INIT_WRITE   4

#define FEM_MESH_OUTPUT 0
#define FEM_MESH_UPDATE 1
#define FEM_MESH_FINALIZE 2
  typedef void (*FEM_Update_mesh_fn)(int userTag);
  typedef void (*FEM_Update_mesh_fortran_fn)(int *userTag);

  typedef void (*FEM_PupFn)(pup_er, void*);

  typedef void (*FEM_Mesh_alloc_fn)(void *param,int *size,int *maxSize);
  
  /* This should be MPI_Comm, but I want it for Fortran too */
  typedef int FEM_Comm_t; 

  /* Initialize the FEM framework (must have called MPI_Init) */
  void FEM_Init(FEM_Comm_t defaultCommunicator);
  void FEM_Done(void);

  /*Utility*/
  int FEM_My_partition(void);
  int FEM_Num_partitions(void);
  double FEM_Timer(void);
  void FEM_Print(const char *str);
  void FEM_Print_partition(void);

/* Mesh manipulation */
#define FEM_MESH_FIRST 1650000000 /*This is the first mesh ID:*/
  /* mesh creation */
  int FEM_Mesh_allocate(void); /* build new mesh */
  int FEM_Mesh_copy(int fem_mesh); /* copy existing mesh */
  void FEM_Mesh_deallocate(int fem_mesh); /* delete this mesh */

  int FEM_Mesh_read(const char *prefix,int partNo,int nParts);
  void FEM_Mesh_write(int fem_mesh,const char *prefix,int partNo,int nParts); 

  int FEM_Mesh_assemble(int nParts,const int *srcMeshes);
  void FEM_Mesh_partition(int fem_mesh,int nParts,int *destMeshes);
  
  int FEM_Mesh_recv(int fromRank,int tag,FEM_Comm_t comm_context);
  void FEM_Mesh_send(int fem_mesh,int toRank,int tag,FEM_Comm_t comm_context);

  int FEM_Mesh_reduce(int fem_mesh,int toRank,FEM_Comm_t comm_context);
  int FEM_Mesh_broadcast(int fem_mesh,int fromRank,FEM_Comm_t comm_context);

  void FEM_Mesh_copy_globalno(int src_mesh,int dest_mesh);
  void FEM_Mesh_print(int fem_mesh);
  
/* Mesh entity codes: (keep in sync with femf.h) */
#define FEM_ENTITY_FIRST 1610000000 /*This is the first entity code:*/
#define FEM_NODE (FEM_ENTITY_FIRST+0) /*The unique node type*/
#define FEM_ELEM (FEM_ENTITY_FIRST+1000) /*First element type (can add the user-defined element type) */
#define FEM_ELEMENT FEM_ELEM /*alias*/
#define FEM_SPARSE (FEM_ENTITY_FIRST+2000) /* First sparse entity (face) type */
#define FEM_EDGE FEM_SPARSE /* alias */
#define FEM_FACE FEM_SPARSE /* alias */
#define FEM_GHOST 10000  /* (entity add-in) Indicates we want the ghost values, not real values */
#define FEM_ENTITY_LAST (FEM_ENTITY_FIRST+3000+FEM_GHOST)

/* Mesh entity "attributes": per-entity data */
#define FEM_DATA   0  /* Backward-compatability routines' solution data: tag 0 */
#define FEM_ATTRIB_TAG_MAX 1000000000 /*Largest allowable user "tag" attribute*/
#define FEM_ATTRIB_FIRST 1620000000 /*This is the first system attribute code: one of*/
#define FEM_CONN   (FEM_ATTRIB_FIRST+1) /* Element-node connectivity (FEM_ELEM or FEM_SPARSE, FEM_INDEX only) */
#define FEM_CONNECTIVITY FEM_CONN /*alias*/

  /* rarely-used external attributes */
#define FEM_SPARSE_ELEM (FEM_ATTRIB_FIRST+2) /* Elements each sparse data record applies to (FEM_SPARSE, 2*FEM_INDEX only) */
#define FEM_COOR   (FEM_ATTRIB_FIRST+3) /* Node coordinates (FEM_NODE, FEM_DOUBLE only) */
#define FEM_COORD FEM_COOR /*alias*/
#define FEM_COORDINATES FEM_COOR /*alias*/
#define FEM_GLOBALNO  (FEM_ATTRIB_FIRST+4) /* Global item numbers (width=1, datatype=FEM_INDEX) */
#define FEM_PARTITION (FEM_ATTRIB_FIRST+5) /* Destination chunk numbers (elements only; width=1, datatype=FEM_INDEX) */
#define FEM_SYMMETRIES (FEM_ATTRIB_FIRST+6) /* Symmetries present (width=1, datatype=FEM_BYTE) */
#define FEM_NODE_PRIMARY (FEM_ATTRIB_FIRST+7) /* This chunk owns this node (nodes only; width=1, datatype=FEM_BYTE) */
#define FEM_CHUNK (FEM_ATTRIB_FIRST+8) /* For Nodes and Elements. Used during ghost creation
to mark the chunk to which a ghost node or element belongs datatype=FEM_INDEX*/
#define FEM_BOUNDARY (FEM_ATTRIB_FIRST+9) /*provides the boundary flag for nodes, elements and sparse elements FEM_INT*/
#define FEM_NODE_ELEM_ADJACENCY (FEM_ATTRIB_FIRST+10) /*node to element adjacency FEM_VAR_INDEX only */
#define FEM_NODE_NODE_ADJACENCY (FEM_ATTRIB_FIRST+11) /*node to node adjacency FEM_VAR_INDEX only */
#define FEM_ELEM_ELEM_ADJACENCY (FEM_ATTRIB_FIRST+12) /*element to element adjacency FEM_VAR_INDEX only */
#define FEM_ELEM_ELEM_ADJ_TYPES (FEM_ATTRIB_FIRST+13) /*stores element types for those element id's listed in 
														FEM_ELEM_ELEM_ADJACENCY, needed when using 
														multiple element types*/
#define FEM_IS_VALID_ATTR (FEM_ATTRIB_FIRST+14) /* Stores a flag(an IDXL_BYTE) for each element or node specifying whether the entity 
										   exists or is valid. It may be 0 whenever a mesh modification occurs that deletes the 
										   corresponding node or element */

#define FEM_MESH_SIZING (FEM_ATTRIB_FIRST+15) /* Target edge length attr. */
#define FEM_ATTRIB_LAST (FEM_ATTRIB_FIRST+16) /*This is the last valid attribute code*/

  /* Specialized routines: */
  void FEM_Mesh_set_conn(int fem_mesh,int entity,
  	const int *conn, int firstItem, int length, int width);
  void FEM_Mesh_get_conn(int fem_mesh,int entity,
  	int *conn, int firstItem, int length, int width);

  void FEM_Mesh_set_data(int fem_mesh,int entity,int attr,
  	const void *data, int firstItem, int length, int datatype,int width);
  void FEM_Mesh_get_data(int fem_mesh,int entity,int attr,
  	void *data, int firstItem, int length, int datatype,int width);
  void FEM_Mesh_conn(int fem_mesh,int entity,
  	int *conn, int firstItem, int length, int width);
  
  int FEM_Mesh_get_length(int fem_mesh,int entity);
  
  /* General purpose routines: */
  void FEM_Mesh_data(int fem_mesh,int entity,int attr,
  	void *data, int firstItem, int length, int datatype,int width);
  void FEM_Mesh_data_layout(int fem_mesh,int entity,int attr,
  	void *data, int firstItem, int length, IDXL_Layout_t layout);
  void FEM_Mesh_data_offset(int fem_mesh,int entity,int attr,
  	void *data, int firstItem, int length, 
	int type,int width, int offsetBytes,int distanceBytes,int skewBytes);
	
  void FEM_Register_array(int fem_mesh,int entity,int attr,
			  void *data, int datatype,int width);

  void FEM_Register_array_layout(int fem_mesh,int entity,int attr, 	
				 void *data, IDXL_Layout_t layout);	
  
  //TODO:add the most important parameter.. the function pointer to the resize function
  void FEM_Register_entity(int fem_mesh,int entity,void *data,int len,int max,FEM_Mesh_alloc_fn fn);	
  
  void FEM_Mesh_set_length(int fem_mesh,int entity,int newLength);
  int FEM_Mesh_get_width(int fem_mesh,int entity,int attr);
  void FEM_Mesh_set_width(int fem_mesh,int entity,int attr,int newWidth);
  int FEM_Mesh_get_datatype(int fem_mesh,int entity,int attr);
  int FEM_Mesh_get_entities(int fem_mesh, int *entities);
  int FEM_Mesh_get_attributes(int fem_mesh,int entity,int *attributes);
  
  const char *FEM_Get_entity_name(int entity,char *storage);
  const char *FEM_Get_attr_name(int attr,char *storage);
  const char *FEM_Get_datatype_name(int datatype,char *storage);

  int FEM_Mesh_is_get(int fem_mesh); /* return 1 if this is a readable mesh */
  int FEM_Mesh_is_set(int fem_mesh); /* return 1 if this is a writing mesh */
  void FEM_Mesh_become_get(int fem_mesh); /* Make this a readable mesh */
  void FEM_Mesh_become_set(int fem_mesh); /* Make this a writing mesh */

  typedef void (*FEM_Userdata_fn)(pup_er p,void *data);
  void FEM_Mesh_pup(int fem_mesh,int dataTag,FEM_Userdata_fn fn,void *data);

/* ghosts and spatial symmetries */
#define FEM_Is_ghost_index(idx) ((idx)<-1)
#define FEM_To_ghost_index(idx) (-(idx)-2)
#define FEM_From_ghost_index(idx) (-(idx)-2)

  void FEM_Add_ghost_layer(int nodesPerTuple,int doAddNodes);
  void FEM_Add_ghost_elem(int elType,int tuplesPerElem,const int *elem2tuple);

  void FEM_Add_ghost_stencil(int nElts,int addNodes,
	const int *ends,const int *adj);
  void FEM_Add_ghost_stencil_type(int elType,int nElts,int addNodes,
	const int *ends,const int *adj2);

  void FEM_Add_elem2face_tuples(int fem_mesh, int elem_type, int nodesPerTuple, int tuplesPerElem,const int *elem2tuple);

  void FEM_Add_linear_periodicity(int nFaces,int nPer,
	const int *facesA,const int *facesB,
	int nNodes,const double *nodeLocs);
  void FEM_Sym_coordinates(int who,double *d_locs);
  
  void FEM_Set_sym_nodes(const int *canon,const int *sym);
  void FEM_Get_sym(int who,int *destSym);
  /**
    Based on shared node communication list, compute 
    FEM_NODE FEM_GLOBALNO and FEM_NODE_PRIMARY
  */
  void FEM_Make_node_globalno(int fem_mesh,FEM_Comm_t comm_context);

/* Communication: see idxlc.h */
  IDXL_Layout_t FEM_Create_simple_field(int base_type,int vec_len);
  IDXL_Layout_t FEM_Create_field(int base_type, int vec_len, int init_offset, 
                       int distance);
  
  IDXL_t FEM_Comm_shared(int fem_mesh,int entity);
  IDXL_t FEM_Comm_ghost(int fem_mesh,int entity);

  void FEM_Get_roccom_pconn_size(int fem_mesh,int *total_len,int *ghost_len);
  void FEM_Get_roccom_pconn(int fem_mesh,const int *paneFmChunk,int *pconn);
  void FEM_Set_roccom_pconn(int fem_mesh,const int *paneFmChunk,const int *src,int total_len,int ghost_len);

  /*Migration */
  int FEM_Register(void *userData,FEM_PupFn _pup_ud);
  void FEM_Migrate(void);
  void *FEM_Get_userdata(int n);
  
  void FEM_Barrier(void);
  
  /* to be provided by the application */
  void init(void);
  void driver(void);
  
  /* Create additional mesh adjacency information */
  void FEM_Mesh_create_node_elem_adjacency(int fem_mesh);
  void FEM_Mesh_create_node_node_adjacency(int fem_mesh);
  void FEM_Mesh_create_elem_elem_adjacency(int fem_mesh);

  void FEM_Print_n2n(int mesh, int nodeid);
  void FEM_Print_n2e(int mesh, int nodeid);
  void FEM_Print_e2e(int mesh, int eid);
  void FEM_Print_e2n(int mesh, int eid);

  /* Create and modify the FEM_IS_VALID Attribute */
  void FEM_Mesh_allocate_valid_attr(int fem_mesh, int entity_type);
  void FEM_set_entity_valid(int mesh, int entityType, int entityIdx);
  void FEM_set_entity_invalid(int mesh, int entityType, int entityIdx);
  int FEM_is_valid(int mesh, int entityType, int entityIdx);
  unsigned int FEM_count_valid(int mesh, int entityType);

  /* Easy set method for coordinates, that may be helpful when creating a mesh */
  void FEM_set_entity_coord2(int mesh, int entityType, int entityIdx, double x, double y);
  void FEM_set_entity_coord3(int mesh, int entityType, int entityIdx, double x, double y, double z);
  

  /* Backward compatability routines: */
  int FEM_Mesh_default_read(void);  /* return mesh used for get calls below */
  int FEM_Mesh_default_write(void); /* return mesh used for set calls below */
  void FEM_Mesh_set_default_read(int fem_mesh);
  void FEM_Mesh_set_default_write(int fem_mesh);
  
  void FEM_Exchange_ghost_lists(int who,int nIdx,const int *localIdx);
  int FEM_Get_ghost_list_length(void);
  void FEM_Get_ghost_list(int *dest);

  void FEM_Update_field(int fid, void *nodes);
  void FEM_Update_ghost_field(int fid, int elTypeOrMinusOne, void *nodes);
  void FEM_Reduce_field(int fid, const void *nodes, void *outbuf, int op);
  void FEM_Reduce(int fid, const void *inbuf, void *outbuf, int op);
  
  void FEM_Read_field(int fid, void *nodes, const char *fname);

  void FEM_Set_node(int nNodes,int doublePerNode);
  void FEM_Set_node_data(const double *data);
  void FEM_Set_elem(int elType,int nElem,int doublePerElem,int nodePerElem);
  void FEM_Set_elem_data(int elType,const double *data);
  void FEM_Set_elem_conn(int elType,const int *conn);
  void FEM_Set_sparse(int uniqueIdentifier,int nRecords,
  	const int *nodes,int nodesPerRec,
  	const void *data,int dataPerRec,int dataType);
  void FEM_Set_sparse_elem(int uniqueIdentifier,const int *rec2elem);

  void FEM_Get_node(int *nNodes,int *doublePerNode);
  void FEM_Get_node_data(double *data);
  void FEM_Get_elem(int elType,int *nElem,int *doublePerElem,int *nodePerElem);
  void FEM_Get_elem_data(int elType,double *data);
  void FEM_Get_elem_conn(int elType,int *conn);
  int  FEM_Get_sparse_length(int uniqueIdentifier); 
  void FEM_Get_sparse(int uniqueIdentifier,int *nodes,void *data);
  
  void FEM_Set_mesh(int nelem, int nnodes, int nodePerElem, int* conn);
  
  int FEM_Get_node_ghost(void);
  int FEM_Get_elem_ghost(int elemType);  

  void FEM_Update_mesh(FEM_Update_mesh_fn callFn,int userValue,int doWhat);
  
  void FEM_Set_partition(int *elem2chunk);


  /* Public functions that modify the mesh */
  int FEM_add_node(int mesh, int* adjacent_nodes=0, int num_adjacent_nodes=0, int *chunks=0, int numChunks=0, int forceShared=0, int upcall=0);
  int FEM_add_element(int mesh, int* conn, int conn_size, int elem_type=0, int chunkNo=-1);
  void FEM_remove_node(int mesh,int node);
  int FEM_remove_element(int mesh, int element, int elem_type=0, int permanent=-1);
  int FEM_purge_element(int mesh, int element, int elem_type=0);
  int FEM_Modify_Lock(int mesh, int* affectedNodes, int numAffectedNodes, int* affectedElts=0, int numAffectedElts=0, int elemtype=0);
  int FEM_Modify_Unlock(int mesh);
  int FEM_Modify_LockN(int mesh, int nodeId, int readLock);
  int FEM_Modify_UnlockN(int mesh, int nodeId, int readLock);
  void FEM_REF_INIT(int mesh, int dim);
  
 

  
  // To help debugging:
  void FEM_Print_Mesh_Summary(int mesh);

/* Routines we wish didn't exist: */
  void FEM_Serial_split(int nchunks);
  void FEM_Serial_begin(int chunkNo);
  
  void FEM_Serial_read(int chunkNo,int nChunks);
  void FEM_Serial_assemble(void);
  
  int FEM_Get_comm_partners(void);
  int FEM_Get_comm_partner(int partnerNo);
  int FEM_Get_comm_count(int partnerNo);
  void FEM_Get_comm_nodes(int partnerNo,int *nodeNos);


/* Routines that no longer exist:
  void FEM_Composite_elem(int newElType);
   -> Replace with IDXL_Create
  void FEM_Combine_elem(int srcElType,int destElType,
	int newInteriorStartIdx,int newGhostStartIdx);  
   -> Replace with IDXL_Combine
*/

#ifdef __cplusplus
}
#endif

#endif


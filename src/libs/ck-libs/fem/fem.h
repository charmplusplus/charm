/*Charm++ Finite Element Framework:
C interface file
*/
#ifndef _CHARM_FEM_H
#define _CHARM_FEM_H
#include "pup_c.h"  /* for pup_er */
#include "idxlc.h"

/* datatypes: keep in sync with femf.h and idxl */
#define FEM_BYTE   IDXL_BYTE
#define FEM_INT    IDXL_INT
#define FEM_REAL   IDXL_REAL
#define FEM_FLOAT FEM_REAL /*alias*/
#define FEM_DOUBLE IDXL_DOUBLE
#define FEM_INDEX_0  IDXL_INDEX_0
#define FEM_INDEX_1  IDXL_INDEX_1 

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


#ifdef __cplusplus
extern "C" {
#endif
  typedef void (*FEM_PupFn)(pup_er, void*);

  /*Attach a new FEM chunk to the existing TCharm array*/
  void FEM_Attach(int flags);

  /*Utility*/
  int FEM_My_partition(void);
  int FEM_Num_partitions(void);
  double FEM_Timer(void);
  void FEM_Done(void);
  void FEM_Print(const char *str);
  void FEM_Print_partition(void);

/* Mesh manipulation */
  /* mesh creation */
  int FEM_Mesh_create_serial(int mesh_sid); /* build new serial mesh, for setting */
  int FEM_Mesh_get_serial(int mesh_sid); /* find existing serial mesh, for getting */
  
  int FEM_Mesh_create(int mesh_context); /* build new parallel mesh, for setting */
  int FEM_Mesh_get_files(const char *baseName,int mesh_context); /* read parallel mesh from files */
  void FEM_Mesh_set_files(int fem_mesh,const char *baseName); /* write parallel mesh to files */
  void FEM_Mesh_destroy(int fem_mesh); /* delete this (parallel or serial) mesh */
  
  int FEM_Mesh_is_get(int fem_mesh); /* return 1 if this is a readable mesh */
  int FEM_Mesh_is_set(int fem_mesh); /* return 1 if this is a writing mesh */
  
  /* mesh update */
#define FEM_MESH_OUTPUT 0
#define FEM_MESH_UPDATE 1
#define FEM_MESH_FINALIZE 2
  typedef void (*FEM_Update_mesh_fn)(int userTag);
  typedef void (*FEM_Update_mesh_fortran_fn)(int *userTag);
  int FEM_Mesh_partition(int mesh_sid); /* partition this serial mesh */
  void FEM_Mesh_assemble(int src_fem_mesh,int dst_mesh_sid,
  	FEM_Update_mesh_fn fn, int userTag); /* reassemble partitioned mesh and call fn */
  

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
#define FEM_ATTRIB_LAST (FEM_ATTRIB_FIRST+10) /*This is the last valid attribute code*/


  /* The basics: */
  void FEM_Mesh_set_conn(int fem_mesh,int entity,
  	const int *conn, int firstItem, int length, int width);
  void FEM_Mesh_get_conn(int fem_mesh,int entity,
  	int *conn, int firstItem, int length, int width);

  void FEM_Mesh_set_data(int fem_mesh,int entity,int attr,
  	const void *data, int firstItem, int length, int datatype,int width);
  void FEM_Mesh_get_data(int fem_mesh,int entity,int attr,
  	void *data, int firstItem, int length, int datatype,int width);
  
  int FEM_Mesh_get_length(int fem_mesh,int entity);
  
  /* Advanced/rarely used: */
  void FEM_Mesh_conn(int fem_mesh,int entity,
  	int *conn, int firstItem, int length, int width);
  void FEM_Mesh_data(int fem_mesh,int entity,int attr,
  	void *data, int firstItem, int length, int datatype,int width);
  void FEM_Mesh_data_layout(int fem_mesh,int entity,int attr,
  	void *data, int firstItem, int length, IDXL_Layout_t layout);
  void FEM_Mesh_data_offset(int fem_mesh,int entity,int attr,
  	void *data, int firstItem, int length, 
	int type,int width, int offsetBytes,int distanceBytes,int skewBytes);
  
  void FEM_Mesh_set_length(int fem_mesh,int entity,int newLength);
  int FEM_Mesh_get_width(int fem_mesh,int entity,int attr);
  void FEM_Mesh_set_width(int fem_mesh,int entity,int attr,int newWidth);
  int FEM_Mesh_get_datatype(int fem_mesh,int entity,int attr);

/* ghosts and spatial symmetries */
#define FEM_Is_ghost_index(idx) ((idx)<-1)
#define FEM_To_ghost_index(idx) (-(idx)-2)
#define FEM_From_ghost_index(idx) (-(idx)-2)

  void FEM_Add_ghost_layer(int nodesPerTuple,int doAddNodes);
  void FEM_Add_ghost_elem(int elType,int tuplesPerElem,const int *elem2tuple);
  
  void FEM_Add_linear_periodicity(int nFaces,int nPer,
	const int *facesA,const int *facesB,
	int nNodes,const double *nodeLocs);
  void FEM_Sym_coordinates(int who,double *d_locs);
  
  void FEM_Set_sym_nodes(const int *canon,const int *sym);
  void FEM_Get_sym(int who,int *destSym);
  

/* Communication: see idxlc.h */
  IDXL_Layout_t FEM_Create_simple_field(int base_type,int vec_len);
  IDXL_Layout_t FEM_Create_field(int base_type, int vec_len, int init_offset, 
                       int distance);
  
  IDXL_t FEM_Comm_shared(int fem_mesh,int entity);
  IDXL_t FEM_Comm_ghost(int fem_mesh,int entity);

  /*Migration */
  int FEM_Register(void *userData,FEM_PupFn _pup_ud);
  void FEM_Migrate(void);
  void *FEM_Get_userdata(int n);
  
  void FEM_Barrier(void);
  
  /* to be provided by the application */
  void init(void);
  void driver(void);
  
/* Backward compatability routines: */
  int FEM_Mesh_default_read(void);  /* return mesh used for get calls below */
  int FEM_Mesh_default_write(void); /* return mesh used for set calls below */
  
  void FEM_Exchange_ghost_lists(int who,int nIdx,const int *localIdx);
  int FEM_Get_ghost_list_length(void);
  void FEM_Get_ghost_list(int *dest);

  void FEM_Add_node(int localIdx,int nBetween,int *betweenNodes);  
  
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


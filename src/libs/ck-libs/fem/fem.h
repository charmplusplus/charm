/*Charm++ Finite Element Framework:
C interface file
*/
#ifndef _FEM_H
#define _FEM_H
#include "converse.h"
#include "pup_c.h"

/* datatypes: keep in sync with femf.h */
#define FEM_FIRST_DATATYPE 11000000 /*first valid FEM datatype (zero is a STUPID value)*/
#define FEM_BYTE   (FEM_FIRST_DATATYPE+0)
#define FEM_INT    (FEM_FIRST_DATATYPE+1) 
#define FEM_REAL   (FEM_FIRST_DATATYPE+2)
#define FEM_FLOAT FEM_REAL /*alias*/
#define FEM_DOUBLE (FEM_FIRST_DATATYPE+3)
#define FEM_INDEX_0  (FEM_FIRST_DATATYPE+4) /*zero-based integer (c-style indexing) */
#define FEM_INDEX_1  (FEM_FIRST_DATATYPE+5) /*one-based integer (Fortran-style indexing) */

/* reduction operations: keep in sync with femf.h */
#define FEM_SUM 0
#define FEM_MAX 1
#define FEM_MIN 2

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
  int FEM_Mesh_create(int mesh_sid1,int mesh_sid2); /* build new mesh */
  int FEM_Mesh_read(const char *baseName); /* read mesh from files */
  void FEM_Mesh_write(int fem_mesh,const char *baseName); /* write mesh to files */
  void FEM_Mesh_destroy(int fem_mesh); /* delete this mesh */
  
  int FEM_Mesh_is_read(int fem_mesh); /* return 1 if this is a readable mesh */
  int FEM_Mesh_is_write(int fem_mesh); /* return 1 if this is a writing mesh */
  
  /* mesh update */
#define FEM_MESH_OUTPUT 0
#define FEM_MESH_UPDATE 1
#define FEM_MESH_FINALIZE 2
  typedef void (*FEM_Update_mesh_fn)(int userTag);
  typedef void (*FEM_Update_mesh_fortran_fn)(int *userTag);
  int FEM_Mesh_partition(int mesh_sid1,int mesh_sid2); /* partition this existing mesh */
  void FEM_Mesh_assemble(int fem_mesh,int mesh_sid1,int mesh_sid2,
  	FEM_Update_mesh_fn fn, int userTag, int how); /* reassemble partitioned mesh */


/* Mesh entity codes: */
#define FEM_ENTITY_FIRST 12000000 /*This is the first entity code:*/
#define FEM_STATIC (FEM_ENTITY_FIRST+1) /* Static (not element or node) mesh data */
#define FEM_NODE (FEM_ENTITY_FIRST+256) /*The only node type*/
#define FEM_ELEM (FEM_ENTITY_FIRST+512) /*First element type (can add the user-defined element type) */
#define FEM_ELEMENT FEM_ELEM /*alias*/
#define FEM_SPARSE (FEM_ENTITY_FIRST+768) /* First "sparse" (face) entity type */
#define FEM_GHOST 2048  /* (entity add-in) Indicates we want the ghost values, not real values */

/* Mesh entity "attributes": per-entity data */
#define FEM_ATTRIB_FIRST 13000000 /*This is the first attribute code:*/
#define FEM_DATA   (FEM_ATTRIB_FIRST+0) /* Solution data */
#define FEM_CONN   (FEM_ATTRIB_FIRST+1) /* Element-node connectivity (FEM_ELEM or FEM_SPARSE, FEM_INDEX only) */
#define FEM_CONNECTIVITY FEM_CONN /*alias*/

  /* rarely-used external attributes */
#define FEM_SPARSE_ELEM (FEM_ATTRIB_FIRST+10) /* Elements each sparse data record applies to (FEM_SPARSE, 2*FEM_INDEX only) */
#define FEM_COOR   (FEM_ATTRIB_FIRST+20) /* Node coordinates (FEM_NODE, FEM_DOUBLE only) */
#define FEM_COORD FEM_COOR /*alias*/
#define FEM_COORDINATES FEM_COOR /*alias*/
#define FEM_GLOBALNO  (FEM_ATTRIB_FIRST+100) /* Global item numbers (no tag; FEM_INDEX only) */
#define FEM_PARTITION (FEM_ATTRIB_FIRST+101) /* Destination chunk numbers (no tag; FEM_INDEX only) */

  /* The basics: */
  void FEM_Mesh_conn(int fem_mesh,int entity,
  	int *conn, int firstItem, int length, int width);
  void FEM_Mesh_data(int fem_mesh,int entity,int tag, 	
  	void *data, int firstItem, int length, int width, int base_type);
  
  int FEM_Mesh_get_length(int fem_mesh,int entity);
  int FEM_Mesh_get_width(int fem_mesh,int entity,int attr,int tag);
  int FEM_Mesh_get_datatype(int fem_mesh,int entity,int attr,int tag);
  
  /* Advanced/rarely used: */
  void FEM_Mesh_attr(int fem_mesh,int entity,int attr, int tag,
  	void *data, int firstItem, int length, int width, int base_type);
  void FEM_Mesh_offset(int fem_mesh,int entity, int attr,int tag,
  	void *data, int firstItem, int length, int width, int base_type, int off, int distL,int distW);
  
  void FEM_Mesh_set_length(int fem_mesh,int entity,int newLength);
  void FEM_Mesh_set_width(int fem_mesh,int entity,int attribute,int tag,int newWidth);
  void FEM_Mesh_set(int fem_mesh,int entity,int attribute,int tag,int newLength,int newWidth);

/* ghosts and spatial symmetries */
  void FEM_Add_ghost_layer(int nodesPerTuple,int doAddNodes);
  void FEM_Add_ghost_elem(int elType,int tuplesPerElem,const int *elem2tuple);
  
  void FEM_Add_linear_periodicity(int nFaces,int nPer,
	const int *facesA,const int *facesB,
	int nNodes,const double *nodeLocs);
  void FEM_Sym_coordinates(int who,double *d_locs);
  
  void FEM_Set_sym_nodes(const int *canon,const int *sym);
  void FEM_Get_sym(int who,int *destSym);
  

/* Communication */
  int FEM_Create_simple_field(int base_type,int vec_len);
  int FEM_Create_field(int base_type, int vec_len, int init_offset, 
                       int distance);
  
  /*FIXME: design a non-blocking send*/
  void FEM_Update_ghost_field(int fid, int elTypeOrMinusOne, void *nodes);
  void FEM_Exchange_ghost_lists(int who,int nIdx,const int *localIdx);
  int FEM_Get_ghost_list_length(void);
  void FEM_Get_ghost_list(int *dest);

  void FEM_Add_node(int localIdx,int nBetween,int *betweenNodes);  

  /*Migration */
  int FEM_Register(void *userData,FEM_PupFn _pup_ud);
  void FEM_Migrate(void);
  void *FEM_Get_userdata(int n);
  
  void FEM_Barrier(void);
  
  /* to be provided by the application */
  void init(void);
  void driver(void);
  
  
/* Backward compatability routines: */
  int FEM_Mesh_default_read(void);  /* return default fem_mesh used for read (get) calls below */
  int FEM_Mesh_default_write(void); /* return default fem_mesh used for write (set) calls below */
  
  void FEM_Update_field(int fid, void *nodes);
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
  void FEM_Composite_elem(int newElType);
  void FEM_Combine_elem(int srcElType,int destElType,
	int newInteriorStartIdx,int newGhostStartIdx);
  
  void FEM_Serial_split(int nchunks);
  void FEM_Serial_begin(int chunkNo);
  
  int FEM_Get_comm_partners(void);
  int FEM_Get_comm_partner(int partnerNo);
  int FEM_Get_comm_count(int partnerNo);
  void FEM_Get_comm_nodes(int partnerNo,int *nodeNos);
  
  int *FEM_Get_node_nums(void);
  int *FEM_Get_elem_nums(void);
  int *FEM_Get_conn(int elemType);



#ifdef __cplusplus
}
#endif

#endif


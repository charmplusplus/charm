/*Charm++ Finite Element Framework:
C interface file
*/
#ifndef _FEM_H
#define _FEM_H
#include "converse.h"
#include "pup_c.h"

/* base types: keep in sync with femf.h */
#define FEM_BYTE   0
#define FEM_INT    1
#define FEM_REAL   2
#define FEM_DOUBLE 3

/* reduction operations: keep in synch with femf.h */
#define FEM_SUM 0
#define FEM_MAX 1
#define FEM_MIN 2

/* element types */
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

  int *FEM_Get_node_nums(void);
  int *FEM_Get_elem_nums(void);
  int *FEM_Get_gonn(int elemType);

  /*Mesh*/
  void FEM_Set_mesh(int nelem, int nnodes, int nodePerElem, int* conn);
  
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
  int  FEM_Get_sparse_length(int uniqueIdentifier); //Returns nRecords
  void FEM_Get_sparse(int uniqueIdentifier,int *nodes,void *data);
  
  void FEM_Add_linear_periodicity(int nFaces,int nPer,
	const int *facesA,const int *facesB,
	int nNodes,const double *nodeLocs);
  void FEM_Sym_coordinates(int elType,double *d_locs);
  
  void FEM_Set_sym_nodes(const int *canon,const int *sym);
  void FEM_Get_sym(int elTypeOrMinusOne,int *destSym);
  
  void FEM_Composite_elem(int newElType);
  void FEM_Combine_elem(int srcElType,int destElType,
	int newInteriorStartIdx,int newGhostStartIdx);

  void FEM_Update_mesh(int callMeshUpdated,int doRepartition);
  
  void FEM_Set_partition(int *elem2chunk);
  void FEM_Serial_split(int nchunks);
  void FEM_Serial_begin(int chunkNo);

  void FEM_Add_ghost_layer(int nodesPerTuple,int doAddNodes);
  void FEM_Add_ghost_elem(int elType,int tuplesPerElem,const int *elem2tuple);

  int FEM_Get_node_ghost(void);
  int FEM_Get_elem_ghost(int elemType);  
  
  int FEM_Get_comm_partners(void);
  int FEM_Get_comm_partner(int partnerNo);
  int FEM_Get_comm_count(int partnerNo);
  void FEM_Get_comm_nodes(int partnerNo,int *nodeNos);

  /*Node update*/
  int FEM_Create_simple_field(int base_type,int vec_len);
  int FEM_Create_field(int base_type, int vec_len, int init_offset, 
                       int distance);
  void FEM_Update_field(int fid, void *nodes);
  void FEM_Update_ghost_field(int fid, int elTypeOrMinusOne, void *nodes);
  void FEM_Reduce_field(int fid, const void *nodes, void *outbuf, int op);
  void FEM_Reduce(int fid, const void *inbuf, void *outbuf, int op);
  void FEM_Read_field(int fid, void *nodes, const char *fname);

  /*Mesh modification*/
  void FEM_Barrier(void);

  void FEM_Add_node(int localIdx,int nBetween,int *betweenNodes);  

  void FEM_Exchange_ghost_lists(int elemType,int nIdx,const int *localIdx);
  int FEM_Get_ghost_list_length(void);
  void FEM_Get_ghost_list(int *dest);

  /*Migration */
  int FEM_Register(void *userData,FEM_PupFn _pup_ud);
  void FEM_Migrate(void);
  void *FEM_Get_userdata(int n);
  
  /* to be provided by the application */
  void init(void);
  void driver(void);
  void finalize(void);
  void mesh_updated(int callMeshUpdated);

#ifdef __cplusplus
}
#endif

#endif


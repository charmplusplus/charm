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

#if FEM_FORTRAN
typedef void (*FEM_PupFn)(pup_er, void*);
#else
typedef void *(*FEM_PupFn)(pup_er, void*);
#endif

#ifdef __cplusplus
extern "C" {
#endif
  /*Utility*/
  int FEM_My_Partition(void);
  int FEM_Num_Partitions(void);
  double FEM_Timer(void);
  void FEM_Done(void);
  void FEM_Print(const char *str);
  void FEM_Print_Partition(void);

  int *FEM_Get_Node_Nums(void);
  int *FEM_Get_Elem_Nums(void);
  int *FEM_Get_Conn(int elemType);

  /*Mesh*/
  void FEM_Set_Mesh(int nelem, int nnodes, int nodePerElem, int* conn);
  
  void FEM_Set_Node(int nNodes,int doublePerNode);
  void FEM_Set_Node_Data(const double *data);
  void FEM_Set_Elem(int elType,int nElem,int doublePerElem,int nodePerElem);
  void FEM_Set_Elem_Data(int elType,const double *data);
  void FEM_Set_Elem_Conn(int elType,const int *conn);

  void FEM_Get_Node(int *nNodes,int *doublePerNode);
  void FEM_Get_Node_Data(double *data);
  void FEM_Get_Elem(int elType,int *nElem,int *doublePerElem,int *nodePerElem);
  void FEM_Get_Elem_Data(int elType,double *data);
  void FEM_Get_Elem_Conn(int elType,int *conn);

  void FEM_Update_Mesh(int callMeshUpdated,int doRepartition);
  
  void FEM_Set_Partition(int *elem2chunk);
  
  int FEM_Get_Comm_Partners(void);
  int FEM_Get_Comm_Partner(int partnerNo);
  int FEM_Get_Comm_Count(int partnerNo);
  void FEM_Get_Comm_Nodes(int partnerNo,int *nodeNos);

  /*Node update*/
  int FEM_Create_Field(int base_type, int vec_len, int init_offset, 
                       int distance);
  void FEM_Update_Field(int fid, void *nodes);
  void FEM_Reduce_Field(int fid, const void *nodes, void *outbuf, int op);
  void FEM_Reduce(int fid, const void *inbuf, void *outbuf, int op);
  void FEM_Read_Field(int fid, void *nodes, const char *fname);

  /*Migration */
  int FEM_Register(void *userData,FEM_PupFn _pup_ud);
  void FEM_Migrate(void);
  void *FEM_Get_Userdata(int n);
  
  /* to be provided by the application */
  void init(void);
  void driver(void);
  void finalize(void);
  void mesh_updated(int callMeshUpdated);

#ifdef __cplusplus
}
#endif

#endif


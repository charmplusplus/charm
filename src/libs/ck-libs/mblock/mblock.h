/*Charm++ Multiblock CFD Framework:
C interface file
*/
#ifndef _MBLK_H
#define _MBLK_H
#include "converse.h"
#include "pup_c.h"

/* base types: keep in sync with mblockf.h */
#define MBLK_BYTE   0
#define MBLK_INT    1
#define MBLK_REAL   2
#define MBLK_DOUBLE 3

/* reduction operations: keep in synch with mblockf.h */
#define MBLK_SUM 0
#define MBLK_MAX 1
#define MBLK_MIN 2

/* return codes */
#define MBLK_SUCCESS 1
#define MBLK_FAILURE 0

/* async comm test */
#define MBLK_DONE 1
#define MBLK_NOTDONE 0

#ifdef __cplusplus
extern "C" {
#endif

  typedef void (*MBLK_PupFn)(pup_er, void*);
  typedef void (*MBLK_BcFn)(void *p1,void *p2,int *start,int *end);

  /* called from init */
  void MBLK_Init(int comm);
  int MBLK_Read(const char *prefix,int nDimensions);
  
  /*Utility*/
  int MBLK_Get_nblocks(int *n);
  int MBLK_Get_myblock(int *m);
  int MBLK_Get_blocksize(int *dims); /*Fetch interior dimensions, in voxels*/
  int MBLK_Get_nodelocs(const int *nodedim,double *nodeloc); 
  double MBLK_Timer(void);
  void MBLK_Print(const char *str);
  void MBLK_Print_block(void);
  
  /* field creation */
  int MBLK_Create_field(int *dimensions,int isVoxel,
      const int base_type, const int vec_len,
      const int offset, const int dist, 
      int *fid);

  /* field update */
  int MBLK_Update_field(const int fid, int ghostWidth, void *grid);
  int MBLK_Iupdate_field(const int fid, int ghostWidth, void *ingrid, void *outgrid);
  int MBLK_Test_update(int *status);
  int MBLK_Wait_update(void);

  /* reduction */
  int MBLK_Reduce_field(const int fid, void *grid, void *out, const int op);
  int MBLK_Reduce(const int fid, void *in, void *out, const int op);

  /* boundary conditions */
  int MBLK_Register_bc(const int bcnum, int ghostWidth, const MBLK_BcFn bcfn);
  int MBLK_Apply_bc(const int bcnum, void *p1,void *p2);
  int MBLK_Apply_bc_all(void *p1,void *p2);

  /*Migration */
  int MBLK_Register(void *userData, MBLK_PupFn _pup_ud, int *rid);
  int MBLK_Migrate(void);
  int MBLK_Get_registered(int rid, void ** block);

#ifdef __cplusplus
}
#endif

#endif


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "fem.h"

extern "C" void
init_(void)
{
  CkPrintf("init_ called\n");
  FILE *fp = fopen("cmesh.dat", "r");
  if(fp==0) { CkAbort("Cannot open cmesh.dat file.\n"); }
  int nelems, nnodes, ctype;
  fscanf(fp, "%d%d%d\n", &nelems, &nnodes, &ctype);
  int i,j;
  int esize = (ctype==FEM_TRIANGULAR)?3:
              ((ctype==FEM_HEXAHEDRAL)?8:
	      4);
  int *conn = new int[nelems*esize];
  for(i=0;i<nelems;i++) {
    for(j=0;j<esize;j++) {
      fscanf(fp,"%d",&conn[i*esize+j]);
    }
  }
  fclose(fp);
  FEM_Set_Mesh(nelems, nnodes, ctype, conn);
}

extern "C" void
driver_(int *nnodes, int *nnums, int *nelems, int *enums, int *npere, int *conn)
{
  FEM_Print_Partition();
  FEM_Done();
}

extern "C" void
finalize_(void)
{
  CkPrintf("finalize_ called\n");
}

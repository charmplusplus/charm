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
  char str[1024];
  char tmpstr[80];
  CkPrintf("[%d] Number of Elements = %d\n", FEM_My_Partition(), *nelems);
  CkPrintf("[%d] Number of Nodes = %d\n", FEM_My_Partition(), *nnodes);
  int i, j;
  sprintf(str, "[%d] List of Elements:\n", FEM_My_Partition());
  for(i=0;i<*nelems;i++) {
    sprintf(tmpstr, "\t%d\n", enums[i]);
    strcat(str,tmpstr);
  }
  CkPrintf("%s", str);
  sprintf(str, "[%d] List of Nodes:\n", FEM_My_Partition());
  for(i=0;i<*nnodes;i++) {
    sprintf(tmpstr, "\t%d\n", nnums[i]);
    strcat(str,tmpstr);
  }
  CkPrintf("%s", str);
  sprintf(str, "[%d] Connectivity:\n", FEM_My_Partition());
  for(i=0;i<*nelems;i++) {
    sprintf(tmpstr, "\t[%d] ", enums[i]);
    strcat(str,tmpstr);
    for(j=0;j<*npere;j++) {
      sprintf(tmpstr, "%d ", nnums[conn[i*(*npere)+j]]);
      strcat(str,tmpstr);
    }
    sprintf(tmpstr, "\n");
    strcat(str,tmpstr);
  }
  CkPrintf("%s", str);
  FEM_Done();
}

extern "C" void
finalize_(void)
{
  CkPrintf("finalize_ called\n");
}

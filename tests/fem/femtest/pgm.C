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

typedef struct _node {
  double val;
} Node;

typedef struct _element {
  double val;
} Element;

extern "C" void
driver_(int *nnodes, int *nnums, int *nelems, int *enums, int *npere, int *conn)
{
  // FEM_Print_Partition();
  Node *nodes = new Node[*nnodes];
  Element *elements = new Element[*nelems];
  int i;
  for(i=0;i<*nnodes;i++) {
    nodes[i].val = 0.0;
    if(nnums[i]==0) { nodes[i].val = 1.0; }
  }
  for(i=0;i<*nelems;i++) { elements[i].val = 0.0; }
  int fid = FEM_Create_Field(FEM_DOUBLE, 1, 0, sizeof(Node));
  int j;
  for(i=0;i<*nelems;i++) {
    for(j=0;j<*npere;j++) {
      elements[i].val += nodes[conn[i*(*npere)+j]].val;
    }
    elements[i].val /= (*npere);
  }
  for(i=0;i<*nnodes;i++) { nodes[i].val = 0.0; }
  for(i=0;i<*nelems;i++) {
    for(j=0;j<*npere;j++) {
      nodes[conn[i*(*npere)+j]].val += elements[i].val;
    }
  }
  FEM_Update_Field(fid, nodes);
  int failed = 0;
  for(i=0;i<*nnodes;i++) {
    if(nnums[i]==0 || nnums[i]==1 || nnums[i]==3 || nnums[i]==4) {
      if(nodes[i].val != 0.25) { failed = 1; }
    } else {
      if(nodes[i].val != 0.0) { failed = 1; }
    }
  }
  if(failed==0) {
    CkPrintf("[chunk %d] update_field test passed.\n", FEM_My_Partition());
  } else {
    CkPrintf("[chunk %d] update_field test failed.\n", FEM_My_Partition());
  }
  double sum = 0.0;
  FEM_Reduce_Field(fid, nodes, &sum, FEM_SUM);
  if(sum==1.0) {
    CkPrintf("[chunk %d] reduce_field test passed.\n", FEM_My_Partition());
  } else {
    CkPrintf("[chunk %d] reduce_field test failed.\n", FEM_My_Partition());
  }
  sum = 1.0;
  FEM_Reduce(fid, &sum, &sum, FEM_SUM);
  if(sum==(double)FEM_Num_Partitions()) {
    CkPrintf("[chunk %d] reduce test passed.\n", FEM_My_Partition());
  } else {
    CkPrintf("[chunk %d] reduce test failed.\n", FEM_My_Partition());
  }
  FEM_Done();
}

extern "C" void
finalize_(void)
{
  CkPrintf("finalize_ called\n");
}

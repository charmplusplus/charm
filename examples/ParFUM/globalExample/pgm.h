#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector2d.h"
#include "ParFUM.h"
#include "ParFUM_internals.h"


//One element's connectivity information
typedef int connRec[3];

// A structure for handling data that may need to be migrated
struct myGlobals {
  int nnodes;
  int nelems;
  int nedges;
  int maxnnodes;
  int maxnelems;
  int maxnedges;
  vector2d *coord;
  connRec *conn;
  int *bounds;
  int *ndata1;
  double *ndata2;
  int *edata1;
};

void pup_myGlobals(pup_er p,myGlobals *g);

// A convenient error function
static void die(const char *str) {
  CkError("Fatal error: %s\n",str);
  CkExit();
}

void resize_nodes(void *data, int *len, int *max);

void resize_elems(void *data, int *len, int *max);

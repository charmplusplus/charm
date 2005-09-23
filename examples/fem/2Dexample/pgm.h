#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "charm++.h"
#include "fem.h"
#include "netfem.h"
#include "vector2d.h"
#include "fem_interpolate.h"


//One element's connectivity information
typedef int connRec[3];
typedef int connRec2[4];

// A structure for handling data that may need to be migrated
struct myGlobals {
  int nnodes;
  int nelems;
  vector2d *coord;
  connRec *conn;

  int nelems2;
  connRec2 *conn2;

  vector2d *R_net, *d, *v, *a;
  
  double *S11, *S22, *S12;
};

void FEM_mesh_smooth(int mesh, int *nodes, int nNodes, int attrNo);

void interpolate(FEM_Interpolate::NodalArgs args,FEM_Mesh *meshP);
//Compute forces on constant-strain triangles:
void CST_NL(const vector2d *coor,const connRec *lm,vector2d *R_net,
	    const vector2d *d,const double *c,
	    int numnp,int numel,
	    double *S11o,double *S22o,double *S12o);

// Prototypes
void advanceNodes(const double dt,int nnodes,const vector2d *coord,
                  vector2d *R_net,vector2d *a,vector2d *v,vector2d *d,bool dampen);

void pup_myGlobals(pup_er p,myGlobals *g);

//The material constants c, as computed by fortran mat_const
// I think the units here are Pascals (N/m^2)
const double matConst[4]={3.692e9,  1.292e9,  3.692e9,  1.200e9 };

//The timestep, in seconds
const double dt=1.0e-9;

// A convenient error function
static void die(const char *str) {
  CkError("Fatal error: %s\n",str);
  CkExit();
}



#define NANCHECK 1 /*Check for NaNs at each timestep*/

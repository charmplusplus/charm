#include "vector2d.h"

//One element's connectivity information
typedef int connRec[3];

//Compute forces on constant-strain triangles:
void CST_NL(const vector2d *coor,const int *lm,vector2d *R_net,
	    const vector2d *d,const double *c,
	    int numnp,int numel,
	    double *S11o,double *S22o,double *S12o);

#include <string>
#include <stdlib.h>
#include "fem.h"

//Coord describes a property with X and Y components
struct Coord
{
  double x;
  double y;
};

// Declaration for class VolElement (Volumetric Elements) 
// These are 6-vertex triangles (each side has an extra vertex)
struct Vol
{
  double s11l[3], s22l[3], s12l[3];//Stress coefficients
};


// Declaration for class CohElement (Cohesive Elements) 
//  These are 6-vertex rectangles (long sides have an extra vertex)
struct Coh
{
  Coord T[3];//Tractions at each sample point
  double sidel[3];//[0]->length of element; 
                //[1],[2] give cosine and sine of orientation
  double Sthresh[3];//The threshold, and damage for this edge
};

// Declaration for class Element (generic elements)
struct Element
{
  int material;          // matlst, matclst, matc
  union {
    Vol v;
    Coh c;
  };
};

  
// Declaration for class Material
struct Material
{
  double c[4], ci[4];          // Elastic stiffness constants & inverses
};

struct VolMaterial : public Material
{
  double e1, e2;               // Young's moduli 
  double g12, g23;             // Shear moduli 
  double xnu12, xnu23;         // Poisson's ratios 
  double rho;                  // density 
  double alpha1, alpha2;       // thermal expansion coefficients 
  double theta;                // principal material direction 
};

struct CohMaterial : public Material
{
  double deltan;           // normal characteristic length
  double deltat;           // tangent characteristic length
  double sigmax;           // max. normal stress
  double taumax;           // max. shearing stress
  double mu;               // coefficient of friction
  double Sinit;            // initial Sthreshold
};
  
struct Node
{
  Coord Rin;             //Internal force
  Coord Rco;             //Cohesive traction load
  Coord xM;              //Mass at this node (xM.x==xM.y, too)
  Coord vel;
  Coord accel;
  Coord disp;
  Coord pos;
  Coord r;
  unsigned char id1, id2;
  unsigned char isbnd;
};

//Global constants
const int    numBoundMax = 1123;   // Maximum number of boundary nodes
const double   g1          = -0.774596669241483;
const double   g3          = 0.774596669241483;
const double   w1          = 0.555555555555555;
const double   w2          = 0.888888888888888;
const double   w3          = 0.555555555555555;
const double   pi          = 3.14159265358979;

struct GlobalData {
  int numNP;               //number of nodal points (numnp)
  int numLST;              //number of LST elements (numlst)
  int numCLST;             //number of LST cohesive elements (numclst)
  int numBound;            //number of boundary nodes w/loads
  int nTime;               //total number of time steps
  int steps;               //ratio of delta and Courant cond
  double delta;            //timestep
  double delta2;           //delta squared times 2????
  int *ts_proportion;      //time step for given constant
  double *proportion;      //load proportionality constant

  int lin;                 //elastic formulation, 0=nl, 1=linear
  int ncoh;                //type of cohesive law, 0=exp., 1=linear
  int nplane;              //type of plane analysis, 0=stress, 1=strain
  int tsintFull;          //output interval in time steps
  int tsintEnergy;        //partial output interval
  int restart;
  int imp;                //Is there impact?
  double voImp;         //Velocity of impactor
  double dImp;             //Displacement
  double aImp;             //Acceleration
  double vImp;             //Velocity
  double xnuImp,eImp,eTop,radiusImp;   //Impact parameters
  double indent;           //indentation of impactor
  int nImp;            //node hit
  double fImp;             //contact force
  double massImp;          //mass

  int numMatVol;           //number of volumetric materials (numat_vol)
  int numMatCoh;           //number of cohesive materials (numat_coh)
  int numProp;             //number of proportionality constants
  VolMaterial *volm;
  CohMaterial *cohm;

  int nn, ne, npere;
  int *nnums, *enums, *conn;
  Node *nodes;
  Element *elements;
  int scoh, ecoh, svol, evol;
};

extern void readFile(GlobalData *gd);
extern void vol_elem(GlobalData *gd);
extern void lst_NL(GlobalData *gd);
extern void lst_coh2(GlobalData *gd);

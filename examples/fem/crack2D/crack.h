/**
 * Main header file for crack propagation code.
 */
#include <string>
#include <stdlib.h>
#include <math.h>
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
  int material; //VolMaterial type
  int conn[6]; // 6 nodes around this element
  double s11l[3], s22l[3], s12l[3];//Stress coefficients
};


// Declaration for class CohElement (Cohesive Elements) 
//  These are 6-vertex rectangles (long sides have an extra vertex)
struct Coh
{
  int material; //CohMaterial type
  int conn[6]; // 6 nodes around this element
  Coord T[3];//Tractions at each sample point
  double sidel[3];//[0]->length of element; 
                //[1],[2] give cosine and sine of orientation
  double Sthresh[3];//The threshold, and damage for this edge
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
  Coord Rin;   //Internal force, from volumetric elements
  Coord Rco;   //Cohesive traction load, from cohesive elements
  Coord xM;    //Mass at this node (xM.x==xM.y)
  Coord vel;
  Coord accel;
  Coord disp;  //Distance this node has moved
  Coord pos;   //Undeformed position of this node
  //FIXME: boundary conditions should be stored in a separate array
  Coord r; //Boundary condition vector
  unsigned char isbnd; //Flag: is this a boundary node?
  unsigned char id1, id2; //Boundary condition flags
};

//Global constants
const double   g1          = -0.774596669241483;
const double   g3          = 0.774596669241483;
const double   w1          = 0.555555555555555;
const double   w2          = 0.888888888888888;
const double   w3          = 0.555555555555555;
const double   pi          = 3.14159265358979;

/// Contains data read from the configuration file that
/// never changes during the run.
struct ConfigurationData {
  /// Simulation control
  int nTime;               //total number of time steps
  int tsintFull;          //output interval in time steps
  int tsintEnergy;        //partial output interval
  int restart;
  
  /// Timestep control
  int steps;               //ratio of delta and Courant cond
  double delta;            //timestep
  double delta2;           //delta squared times 2????
  int numProp;             //number of proportionality constants
  int *ts_proportion;      //1-based time step to apply proportionality constant at
  double *proportion;      //boundary load proportionality constant

  /**
   Return the fraction of the boundary conditions (prop) and time-rate of
   change of boundary conditions (slope) to apply during this 0-based timestep.  
   Applies linear interpolation to the ts_proportion and proportion arrays, above.
  */
  void getBoundaryConditionScale(int timestep,double *prop,double *slope) const;

  /// Material formulation
  int lin;                 //elastic formulation, 0=nl, 1=linear
  int ncoh;                //type of cohesive law, 0=exp., 1=linear
  int nplane;              //type of plane analysis, 0=stress, 1=strain
  
  /// Material properties
  int numMatVol;           //number of volumetric materials (numat_vol)
  int numMatCoh;           //number of cohesive materials (numat_coh)
  VolMaterial *volm;  // Properties of volumetric materials
  CohMaterial *cohm;  // Properties of cohesive materials
  
  /// "Impact": special boundary condition on just one node
  int imp;                //Is there impact?
  double voImp;         //Velocity of impactor
  double dImp;             //Displacement
  double aImp;             //Acceleration
  double vImp;             //Velocity
  double xnuImp,eImp,eTop,radiusImp;   //Impact parameters
  double indent;           //indentation of impactor
  double fImp;             //contact force
  double massImp;          //mass
};

extern ConfigurationData config;
void readConfig(const char *configFile,const char *meshFile);

/// This structure describes the nodes and elements of a mesh:
struct MeshData {
  int nn;               //number of nodal points (numnp)
  int ne;              //number of LST elements (numlst)
  int nc;             //number of LST cohesive elements (numclst)
  int numBound;            //number of boundary nodes w/loads
  int nImp;            //node hit by impactor

  Node *nodes;
  Vol *vols;
  Coh *cohs;
};

//Serial mesh routines: in mesh.C
void readMesh(MeshData *mesh,const char *meshFile);
void setupMesh(MeshData *mesh); //Initialize newly-read mesh
void deleteMesh(MeshData *mesh); //Free storage allocated in mesh

//Parallel mesh routines: in fem_mesh.C
void sendMesh(MeshData *mesh,int fem_mesh);
void recvMesh(MeshData *mesh,int fem_mesh);
extern "C" void pupMesh(pup_er p,MeshData *mesh); //For migration


//Node physics: in node.C
struct NodeSlope {
  double prop; //Proportion of boundary conditions to apply
  double slope; //Slope of prop
};
void nodeSetup(NodeSlope *sl);
void nodeBeginStep(MeshData *mesh);
void nodeFinishStep(MeshData *mesh, NodeSlope *sl,int tstep);

//Element physics: in lst_NL.C, lst_coh2.C
extern void lst_NL(MeshData *mesh);
extern void lst_coh2(MeshData *mesh);


void crack_abort(const char *why);

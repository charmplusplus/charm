/**
Shape function and interpolation support for GenericElementType.

Written by Mike Campbell, 2003.
Modified by Orion Lawlor, 2004.
*/

#include <charm.h> /* for CkAbort */
#include <vector>
#include <list>
using std::list;
using std::vector;

#include <GenericElement.h>

/* Numerical fudge factors */
#ifndef TOL
#define TOL 1e-9
#endif
/// Used to test element inclusion-- be generous on the boundaries
const double LTOL = 0.0 - TOL;
const double HTOL = 1.0 + TOL;

/************* Element Shape Functions **********/

// Populate SF with the shape function of the element
void 
GenericElement::shape_func(const CVector &nc,double SF[]) const
{
  switch (_size) {
    // Tets are 4 nodes or 10
  case 4: {
    SF[0] = 1. - nc.x - nc.y - nc.z;
    SF[1] = nc.x;
    SF[2] = nc.y;
    SF[3] = nc.z;
    break;
  }
  case 10: {
    const double xi = nc.x;
    const double eta = nc.y;
    const double zeta = nc.z;
    const double alpha = (1. - xi - eta - zeta);
    SF[0] = alpha*(1. - 2.*(xi + eta + zeta));
    SF[1] = xi *(2.* xi - 1.);
    SF[2] = eta *(2. * eta - 1.);
    SF[3] = zeta *(2. * zeta - 1.);
    SF[4] = 4.* xi * alpha;
    SF[5] = 4.* eta * alpha;
    SF[6] = 4.* zeta * alpha;
    SF[7] = 4. * xi * eta;
    SF[8] = 4. * eta * zeta;
    SF[9] = 4. * xi * zeta;
    break;
  }
    // Hex's are 8 nodes or 20
  case 8: {
    const double xi = nc.x;
    const double xi_minus = 1. - xi;
    const double eta = nc.y;
    const double eta_minus = 1. - eta;
    const double zeta = nc.z;
    const double zeta_minus = 1. - zeta;
    SF[0] = xi_minus * eta_minus * zeta_minus;
    SF[1] = xi * eta_minus * zeta_minus;
    SF[2] = xi * eta * zeta_minus;
    SF[3] = xi_minus * eta * zeta_minus;
    SF[4] = xi_minus * eta_minus * zeta;
    SF[5] = xi * eta_minus * zeta;
    SF[6] = xi * eta * zeta;
    SF[7] = xi_minus * eta * zeta;
    break;
  }
  default: 
    CkAbort("GenericElement::shape_func:Error: unkown element type.");
  }
}

// Populate dSF with the derivative of the shape function for the element
void
GenericElement::dshape_func(const CVector &nc,double dSF[][3]) const
{
  switch (_size) {
  case 4: {
    dSF[0][0] = -1;  dSF[0][1] = -1;  dSF[0][2] = -1;
    dSF[1][0] =  1;  dSF[1][1] =  0;  dSF[1][2] =  0;
    dSF[2][0] =  0;  dSF[2][1] =  1;  dSF[2][2] =  0;
    dSF[3][0] =  0;  dSF[3][1] =  0;  dSF[3][2] =  1;
    break;
  }
  case 10:{
    const double xi = nc.x;
    const double eta = nc.y;
    const double zeta = nc.z;
    const double alpha = (1. - xi - eta - zeta);
    dSF[0][0] = (4.*(xi+eta+zeta)-3.);      dSF[0][1] = dSF[0][0];                 dSF[0][2] = dSF[0][0];
    dSF[1][0] = 4.*xi - 1.;                 dSF[1][1] = 0;                         dSF[1][2] = 0;
    dSF[2][0] = 0;                          dSF[2][1] = 4.*eta - 1.;               dSF[2][2] = 0;
    dSF[3][0] = 0;                          dSF[3][1] = 0;                         dSF[3][2] = 4.*zeta - 1.;
    dSF[4][0] = 4.*(alpha - xi);            dSF[4][1] = -4.*xi;                    dSF[4][2] = -4.*xi;
    dSF[5][0] = -4.*eta;                    dSF[5][1] = 4.*(alpha - eta);          dSF[5][2] = -4.*eta;
    dSF[6][0] = -4.*zeta;                   dSF[6][1] = -4.*zeta;                  dSF[6][2] = 4.*(alpha - zeta);
    dSF[7][0] = 4.*eta;                     dSF[7][1] = 4.*xi;                     dSF[7][2] = 0;
    dSF[8][0] = 0;                          dSF[8][1] = 4.*zeta;                   dSF[8][2] = 4.*eta;
    dSF[9][0] = 4.*zeta;                    dSF[9][1] = 0;                         dSF[9][2] = 4.*xi;
    break;
  }    
  case 8: {
    const double xi = nc.x;
    const double xi_minus = 1. - xi;
    const double eta = nc.y;
    const double eta_minus = 1. - eta;
    const double zeta = nc.z;
    const double zeta_minus = 1. - zeta;
    dSF[0][0] = -1.*eta_minus*zeta_minus;  dSF[0][1] = -1.*xi_minus*zeta_minus;  dSF[0][2] = -1.*xi_minus*eta_minus;
    dSF[1][0] = eta_minus*zeta_minus;      dSF[1][1] = -1.*xi*zeta_minus;        dSF[1][2] = -1.*xi*eta_minus;
    dSF[2][0] = eta*zeta_minus;            dSF[2][1] = xi*zeta_minus;            dSF[2][2] = -1.*xi*eta;
    dSF[3][0] = -1.*eta*zeta_minus;        dSF[3][1] = xi_minus*zeta_minus;      dSF[3][2] = -1.*xi_minus*eta;
    dSF[4][0] = -1.*eta_minus*zeta;        dSF[4][1] = -1.*xi_minus*zeta;        dSF[4][2] = xi_minus*eta_minus;
    dSF[5][0] = eta_minus*zeta;            dSF[5][1] = -1.*xi*zeta;              dSF[5][2] = xi*eta_minus;
    dSF[6][0] = eta*zeta;                  dSF[6][1] = xi*zeta;                  dSF[6][2] = xi*eta;
    dSF[7][0] = -1.*eta*zeta;              dSF[7][1] = xi_minus * zeta;          dSF[7][2] = xi_minus*eta;
    break;
  }
  default:
    CkAbort("GenericElement::dshape_func:error: Unknown element type.");
  }
}

// Populate J with the jacobian for the point p having natural coordinates nc.  
void
GenericElement::jacobian(const CPoint p[],const CVector &nc,CVector J[]) const
{
  switch(_size){
  case 4: {
    J[0] = p[1] - p[0];
    J[1] = p[2] - p[0];
    J[2] = p[3] - p[0];
    break;
  }
  case 10: {
    const double xi = nc.x;
    const double eta = nc.y;
    const double zeta = nc.z;
    const double alpha = (1. - xi - eta - zeta);
    CPoint P(p[0]*(4.*(xi+eta+zeta)-3.));
    J[0] = ((p[9]-p[6])*4.*zeta)+((p[7]-p[5])*4.*eta)+
      (p[4]*(4.*(alpha-xi))+p[1]*(4.*xi-1.)+P);
    J[1] = ((p[8]-p[6])*4.*zeta)+((p[7]-p[4])*4.*xi)+
      (p[5]*(4.*(alpha-eta))+p[2]*(4.*eta-1.)+P);
    J[2] = ((p[9]-p[4])*4.*xi)+((p[8]-p[5])*4.*eta)+
      (p[6]*(4.*(alpha-zeta))+p[3]*(4.*zeta-1.)+P);
    break;
  }
  case 8: {
    const double xi = nc.x;
    const double xi_minus = 1. - xi;
    const double eta = nc.y;
    const double eta_minus = 1. - eta;
    const double zeta = nc.z;
    const double zeta_minus = 1. - zeta;
    J[0] = ((p[6]-p[7])*eta*zeta)+((p[5]-p[4])*eta_minus*zeta)+
      ((p[2]-p[3])*eta*zeta_minus)+((p[1]-p[0])*eta_minus*zeta_minus);
    J[1] = ((p[7]-p[4])*xi_minus*zeta)+((p[6]-p[5])*xi*zeta)+
      ((p[3]-p[0])*xi_minus*zeta_minus)+((p[2]-p[1])*xi*zeta_minus);
    J[2] = ((p[7]-p[3])*xi_minus*eta)+((p[6]-p[2])*xi*eta)+
      ((p[5]-p[1])*xi*eta_minus)+((p[4]-p[0])*xi_minus*eta_minus);
    break;
  }
  default:
    CkAbort("GenericElement::jacobian:Error: Cannot handle this element size (yet).");
  }
}

/// Interpolate nValuesPerNode doubles from src to dest.
void 
GenericElement::interpolate_natural(int nValuesPerNode,
  		   const ConcreteElementNodeData &src, // Source element
		   const CVector &nc,
		   double *dest) const
{
  const double xi = nc.x;
  const double eta = nc.y;
  const double zeta = nc.z;
  for (int i=0;i<nValuesPerNode;i++) // loop over data values
  {
    double f[maxSize]; // data for each of our nodes
    for (int n=0;n<_size;n++) f[n]=src.getNodeData(n)[i];
    switch(_size) {
    case 4: {
      dest[i] = f[0]+(((f[1]-f[0])*xi) + ((f[2]-f[0])*eta) + ((f[3] - f[0])*zeta));
      break;
    }
    case 10: {
      const double alpha = (1.-xi-eta-zeta);
      dest[i] = (alpha*(1.-2.*(xi+eta+zeta))*f[0] +
   	   xi*(2.*xi-1.)*f[1] +
   	   eta*(2.*eta-1.)*f[2] +
   	   zeta*(2.*zeta-1.)*f[3] +
   	   4.*xi*alpha*f[4] +
   	   4.*eta*alpha*f[5] +
   	   4.*zeta*alpha*f[6] +
   	   4.*xi*eta*f[7] +
   	   4.*eta*zeta*f[8] +
   	   4.*zeta*xi*f[9]);
      break;
    }
    case 8: {
      const double xi = nc.x;
      const double xi_minus = 1. - xi;
      const double eta = nc.y;
      const double eta_minus = 1. - eta;
      const double zeta = nc.z;
      const double zeta_minus = 1. - zeta;
      dest[i] = (xi_minus*eta_minus*zeta_minus*f[0] +
   	   xi*eta_minus*zeta_minus*f[1] +
   	   xi*eta*zeta_minus*f[2] +
   	   xi_minus*eta*zeta_minus*f[3] +
   	   xi_minus*eta_minus*zeta*f[4] +
   	   xi*eta_minus*zeta*f[5] +
   	   xi*eta*zeta*f[6] +
   	   xi_minus*eta*zeta*f[7]);
      break;
    }
    default:
      CkAbort("interpolate::error Cannot handle this element type (yet).");
    }
  }
}

/// Transpose these three vectors as a (row or column) matrix
void 
Transpose(CVector matrix[])
{
  CVector tpose[3];
  
  tpose[0]=CVector(matrix[0].x,matrix[1].x,matrix[2].x);
  tpose[1]=CVector(matrix[0].y,matrix[1].y,matrix[2].y);
  tpose[2]=CVector(matrix[0].z,matrix[1].z,matrix[2].z);
  matrix[0]=tpose[0];
  matrix[1]=tpose[1];
  matrix[2]=tpose[2];
}

// Evaluates the shape function and jacobian at a given point
void
GenericElement::shapef_jacobian_at(const CPoint &p,CVector &natc,
				   const ConcreteElement &e, // Source element
				   CVector &fvec,CVector fjac[]) const
{  
  CPoint P[maxSize];
  double SF[maxSize];
  this->shape_func(natc,SF);
  fvec=-p;
  for(int i = 0;i < _size;i++){
    P[i]=e.getNodeLocation(i);
    fvec += SF[i]*P[i];
  }
  this->jacobian(P,natc,fjac);
  Transpose(fjac);
}



/******************* Point Location ********************/


bool
NewtonRaphson(CVector &natc,
	      const GenericElement &el,
	      const ConcreteElement &e, // Source element
	      const CPoint &point);
bool 
LUDcmp(CVector a[], 
       int indx[]);

void 
LUBksb(CVector a[],
       int indx[], 
       CVector &b);


// Return true and the natural coordinates if this point lies in this element.
bool
GenericElement::element_contains_point(const CPoint &p, //    Target Mesh point
				   const ConcreteElement &e, // Source element
				   CVector &natc) const // Returns Targ nat
{
  // guess at natural coordinates of point: center of element
  if(_size == 4 || _size == 10)
    natc=CVector(.25,.25,.25);
  else if (_size == 8 || _size == 20)
    natc=CVector(.5,.5,.5);
  else{
    CkAbort("GenericElement::element_contains_point: Error: Cannot handle this element type. (yet)");
  }
  
  // Solve for the natural coordinates using non-linear newton-raphson 
  if(!NewtonRaphson(natc,*this,e,p)){
    CkAbort("GenericElement::global_find_point_in_mesh: error NewtonRaphson failed.");
  }
  
  // Make sure natural coordinates lie in unit cube:
  if(natc[0] >= LTOL && natc[0] <= HTOL &&
     natc[1] >= LTOL && natc[1] <= HTOL &&
     natc[2] >= LTOL && natc[2] <= HTOL)
  {
    if(_size == 4 || _size == 10){
      // Natural coordinates must sum to less than one.
      if((natc[0]+natc[1]+natc[2]) <= HTOL){
        return (true);
      }
    }
    else if(_size == 8 || _size == 20) { 
      // Everything in unit cube is OK.
      return(true);
    } 
  }
  return false; // Point's natural coordinates do not lie in unit cube
}

// Newton-Raphson method, customized for using the GeoPrimitives
// and computational mesh constructs
//
// Parameters: 
// double natc[] = initial guess at the natural coordinates 
// unsigned int elnum = index of the mesh element to use
// const ElementConnectivity &ec = The element connectivity for the mesh
// const NodalCoordinates &nc = The nodal coordinates for the mesh
// const CPoint &point = the point at which we wish a solution
//
bool
NewtonRaphson(CVector &natc,
	      const GenericElement &el,
	      const ConcreteElement &e, // Source element
	      const CPoint &point) 
{
  int k,i;
  int ntrial = 4; // Number of iterations before giving up
  double errx,errf,d;
  int indx[3];
  CVector p;
  CVector fvec;
  CVector fjac[3];
  for (k=0;k<ntrial;k++) {
    el.shapef_jacobian_at(point,natc,e,fvec,fjac);
    errf=0.0;
    for (i=0;i<3;i++) 
      errf += fabs(fvec[i]);
    if (errf <= TOL)
      return (true);
    p = -1.0 * fvec;
    if(!LUDcmp(fjac,indx)){
      // cerr << "NewtonRaphson::error: LUDcmp failed." << endl;
      return(false);
    }
    LUBksb(fjac,indx,p);
    errx=0.0;
    for (i=0;i<3;i++) 
      errx += fabs(p[i]);
    natc += p;
    if (errx <= TOL)
      return (true);
  }
  // cerr << "NewtonRaphson::warning: reached maximum iterations" << endl;
  return (true);
}


// LU Decomp
#define TINY 1.0e-20
bool
LUDcmp(CVector a[], int indx[])
{
  int i,imax,j,k;
  double big,dum,sum,temp,d;
  CVector vv;
  
  
  
  for (i=0;i<3;i++) {
    big=0.0;
    for (j=0;j<3;j++)
      if ((temp=fabs(a[i][j])) > big) big=temp;
    if (big == 0.0){
      // cerr << "LUDcmp::error: Singular matrix" << endl;
      return(false);
    }
    vv[i]=1.0/big;
  }
  for (j=0;j<3;j++) {
    for (i=0;i<j;i++) {
      sum=a[i][j];
      for (k=0;k<i;k++) 
	sum -= a[i][k]*a[k][j];
      a[i][j]=sum;
    }
    big=0.0;
    for (i=j;i<3;i++) {
      sum=a[i][j];
      for (k=0;k<j;k++)
	sum -= a[i][k]*a[k][j];
      a[i][j]=sum;
      if ( (dum=vv[i]*fabs(sum)) >= big) {
	big=dum;
	imax=i;
      }
    }
    if (j != imax) {
      for (k=0;k<3;k++) {
	dum=a[imax][k];
	a[imax][k]=a[j][k];
	a[j][k]=dum;
      }
      vv[imax]=vv[j];
    }
    indx[j]=imax;
    if (a[j][j] == 0.0) a[j][j]=TINY;
    if (j != 3) {
      dum=1.0/(a[j][j]);
      for (i=j+1;i<3;i++) 
	a[i][j] *= dum;
    }
  }
  return (true);
}
#undef TINY

// LU Back substitution
void
LUBksb(CVector a[],int indx[], CVector &b)
{
  int i,ii=0,ip,j;
  double sum;
  
  for (i=0;i<3;i++) {
    ip=indx[i];
    sum=b[ip];
    b[ip]=b[i];
    if (ii){
      for (j=ii;j<=i-1;j++) 
	sum -= a[i][j]*b[j];
    }
    else if (sum) 
      ii=i;
    b[i]=sum;
  }
  for (i=2;i>=0;i--) {
    sum=b[i];
    for (j=i+1;j<3;j++) 
      sum -= a[i][j]*b[j];
    b[i]=sum/a[i][i];
  }
}


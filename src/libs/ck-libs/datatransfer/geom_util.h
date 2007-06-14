#ifndef GEOM_UTIL_H
#define GEOM_UTIL_H

#include        <iostream>
#include        <stdio.h>
#include        <stdlib.h>
#include        <math.h>
#include        <vector>
#include        <mapbasic.h>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

using namespace std;

typedef MAP::Vector_3<double> Vec3D;
typedef MAP::Vector_2<double> tPointd;


typedef std::vector<tPointd> tPolygond;
typedef enum { Pin, Qin, Unknown } tInFlag;

/***********************************************************/
// Simple 3D ray class
/***********************************************************/

class Ray3D
{

public:

  Vec3D src;    // src point of the ray
  Vec3D delta;  // direction of the ray

  Ray3D(){}; 

  // Construct a ray by providing a source point
  // and desitnation point.
  Ray3D(Vec3D & s, Vec3D & d)
  {
    src = s;
    delta = d-s;
  };
  
  void set(double* s , double* d)
  {
    src[0] = s[0];
    src[1] = s[1];
    src[2] = s[2];
    delta[0] = d[0]-s[0]; 
    delta[1] = d[1]-s[1]; 
    delta[2] = d[2]-s[2];
  }
  
  Vec3D get_point(double d)
  {
    return Vec3D(src[0] + (d*delta[0]),
		 src[1] + (d*delta[1]),
		 src[2] + (d*delta[2]));
  }

/*   void draw() */
/*   { */
/*     glPushMatrix(); */
/*     glLoadIdentity(); */
/*     glDisable(GL_LIGHTING); */
/*     glColor3f(1.0,1.0,0.0); */
/*     glPointSize(10.0); */
/*     glBegin(GL_POINTS); */
/*     glVertex3f(src[0],src[1],src[2]); */
/*     glVertex3f(src[0]+delta[0],src[1]+delta[1],src[2]+delta[2]); */
/*     glEnd(); */
/*     glBegin(GL_LINES); */
/*     glVertex3f(src[0],src[1],src[2]); */
/*     glVertex3f(src[0]+delta[0],src[1]+delta[1],src[2]+delta[2]); */
/*     glEnd();glPopMatrix(); */
/*     //glEnable(GL_LIGHTING); */
/*   } */
};

//**************************************************/
// Simple 3D plane class
//**************************************************/

class Plane3D
{

 public:
  Vec3D p; // point on plane
  Vec3D n; // normal of the plane
  Vec3D u; // u,v are orthogonal unit vectors
  Vec3D v; // defining a local 2D frame 

  Plane3D(){};
  
  // Construct a plane from a 3D triangle
  void from_tri(double tri[][3])
  {
    Vec3D t1,t2;
 
    p[0] = tri[0][0];
    p[1] = tri[0][1];
    p[2] = tri[0][2];    
    t1[0] = tri[1][0] - p[0];
    t1[1] = tri[1][1] - p[1];
    t1[2] = tri[1][2] - p[2];
    t2[0] = tri[2][0] - p[0];
    t2[1] = tri[2][1] - p[1];
    t2[2] = tri[2][2] - p[2];
    n = (Vec3D) MAP::Vector_3<double>::cross_product(t1,t2);
    n.normalize();

    Vec3D u,v;
    u[0] = tri[1][0]-p[0];
    u[1] = tri[1][1]-p[1];
    u[2] = tri[1][2]-p[2];
    v =  (Vec3D) MAP::Vector_3<double>::cross_product(n,u);
    set_local_frame(u,v);
  }

  double intersect_ray(Ray3D & r)
  {
    double d,t;
    d = -(p*n);
    t = -((n*r.src) +d)/(n*r.delta);
    return t;
  }

  void set_local_frame(Vec3D & ul, Vec3D & vl)
    {
      u = ul; u.normalize(); 
      v = vl; v.normalize();    
    }

  // map assumes p3d is a point in R3 that is on this 
  // plane, and maps to the equivalent 2D point in the
  // local frame of the plane
  void map(Vec3D & p3d, tPointd & p2d)
    {
      Vec3D tp3d = p3d-p;
      p2d[0] = tp3d*u;
      p2d[1] = tp3d*v;
    }

  void unmap(Vec3D & p3d, tPointd & p2d)
    {
      p3d = p + p2d[0]*u + p2d[1]*v;
    }

/*   //#ifdef HAVE_OPENGL */
/*   void draw() */
/*   { */
/*     glPushMatrix(); */
/*     glLoadIdentity(); */
/*     glDisable(GL_LIGHTING); */
/*     glColor3f(1.0,0.0,0.0); */
/*     glPointSize(10.0); */
/*     glBegin(GL_POINTS); */
/*     glVertex3f(p[0],p[1],p[2]); */
/*     glEnd(); */
/*     glBegin(GL_LINES); */
/*     glVertex3f(p[0],p[1],p[2]); */
/*     glVertex3f(p[0]+n[0],p[1]+n[1],p[2]+n[2]); */
/*     glEnd();glPopMatrix(); */
/*     //glEnable(GL_LIGHTING); */
/*   } */
/*   //#endif */


};

//*******************************************************************/
// Helper function to erase consecutive duplicates in a ring vector
template<typename T>
void erase_consecutive_dups(vector<T> & v)
{ 
  int d =0;
  typename vector<T>::iterator i = unique(v.begin(), v.end());
  v.erase(i,v.end());
  if (v[0] == v[v.size()-1]) v.erase(v.end()-1,v.end());
}

//********************************************************************/
//Function prototypes.
bool    point_in_convpoly3D(Vec3D & p, vector<Plane3D> & P);
bool    point_in_convpoly2D(tPointd & p, tPolygond & P);
bool    poly_in_convpoly2D(tPolygond & P1, tPolygond & P2);
void    radial_sort2D(tPolygond & P); 
double  tri_prism_X(double tri[][3], double prism[][3], vector<Vec3D>  & xpoly);
int     convex_poly_X( tPolygond & P, tPolygond & Q, tPolygond & I);

int	area_sign( tPointd & a, tPointd & b, tPointd & c );
char    seg_seg_int( tPointd & a, tPointd & b, tPointd & c, 
		     tPointd & d, tPointd & p, tPointd & q );
char    parallel_int( tPointd & a, tPointd & b, tPointd & c, 
		      tPointd & d, tPointd & p, tPointd & q );
bool    between( tPointd & a, tPointd & b, tPointd & c );
bool    left_on( tPointd & a, tPointd & b, tPointd & c );
bool    left( tPointd & a, tPointd & b, tPointd & c );
tInFlag in_out( tPointd & p, tInFlag inflag, int aHB, int bHA, tPolygond & I );
int     advance( int a, int *aa, int n, bool inside, tPointd & v , tPolygond & I);
void	shared_seg( tPointd & p, tPointd & q, tPolygond & I );


#endif


#include "geom_util.h"

//FIX inline functions 

bool point_in_convpoly3D(Vec3D & p, vector<Plane3D> & P)
{
  Vec3D v;
  for(int i=0;i<P.size();i++)
    {
      v = p-P[i].p;
      if ((P[i].n*v) > 0)
	return false;
    }
  return true;
}

// Sort a set of points forming a 2D convex polygon into
// counterclockwise order. Uses a modified bubble-sort 
// (it's OK, the number of points will be small so asymptotics 
// don't matter here).
void radial_sort2D(tPolygond &P)
{ int i, j, flag = 1;    // set flag to 1 to begin initial pass
  tPointd temp;             // holding variable
  int l = P.size(); 
  // find lowest, rightest point
  short p=0;
  
  for(i=1;i<P.size();i++)
    {
      
      if (P[i][1] < P[p][1])
	p = i;
      else if  (P[i][1] == P[p][1])
	{
	  if (P[i][0] > P[p][0])
	    p = i;
	}
    }
 
  //...bubble sort
  for(i = 1; (i <=l) && flag; i++)
     {
       flag = 0;
       for (j=0; j <(l-1); j++)
         {
	   if (  left(P[p],P[j+1],P[j]))      //P[j] is left of line P[p], P[j+1]
              { 
                    temp = P[j];             // swap elements
                    P[j] = P[j+1];
                    P[j+1] = temp;
                    flag = 1;    
               }
          }
     }
     return;  
}

inline double poly_area2D(tPolygond &P)
{
     double area = 0;
     int   i, j, k,n; 
     n=P.size();
     for (i=1, j=2, k=0; i<=n; i++, j++, k++) {
       area += P[i%n][0] * (P[j%n][1] - P[k%n][1]);
     }
     return area / 2.0;
}

bool point_in_convpoly2D(tPointd & p, tPolygond & P)
{
  tPointd ne;
  tPointd v;
  int last = P.size()-1;

  for(int i=0;i<last;i++)
    {
      ne[0] = -1.0*P[i+1][1] - P[i][1];
      ne[1] = P[i+1][0] - P[i][0];
      v = p-P[i];
      if ((ne*v) > 0)
	return false;
    }

  ne[0] = -1.0*P[0][1] - P[last][1];
  ne[1] = P[0][0] - P[last][0];
  v = p-P[last];
  if ((ne*v) > 0)
	return false;

  return true;
}

bool  poly_in_convpoly2D(tPolygond & P1, tPolygond & P2)
{
  for(int i=0;i<P1.size();i++)
    {
      if (point_in_convpoly2D(P1[i],P2) == false)
	{
	  return false;
	}
    }
  return true;
}

// //**********************************************************************/
// Function:  tri_prism_X
//
// Intersect a 3D triangle with a 3D, 5-sided prism.
//
// Returns: Area of intersection 
//
// Prameters: tri  3 vertices of a triangle ordered counter-clockwise
//                 according to the right-hand rule:
//
//                                  v0
//                                  /\
//                                 /  \
//                                v1--v2
//
//                 Vertex i coordinates are in tri[i][0], tri[i][1], 
//                 tri[i][2]
//
//
//             prism   6 vertices ordered around two triangular faces 
//                     counter-clockwise
//       
//                                  p0           p3
//                                  /\          /\
//                                 /  \        /  \
//                                p1--p2      p4--p5  
//     
//                     with edges [p0,p3], [p1,p5], [p2,p4]
//                     Vertex i coordinates are in prism[i][0],...,
//                     prism[i][3].
//
//*********************************************************************/

double tri_prism_X(double tri[][3], double prism[][3], vector<Vec3D>  & xpoly)
{
  tPolygond tpoly, ppoly, xpoly2D;
  vector<Vec3D>  xpoly3, testpoly;
  Vec3D temp;
  double area;
  std::vector<Vec3D> trace; //Intersection points of the prism with
                             //the plane defined by the triangle

  // Compute plane of the triangle
  Plane3D P;
    
  P.from_tri(tri);

  const int n_edges = 9;
  Ray3D ray[n_edges];
  ray[0].set(prism[0], prism[1]);
  ray[1].set(prism[0], prism[2]);
  ray[2].set(prism[1], prism[2]);

  ray[3].set(prism[3], prism[4]);
  ray[4].set(prism[3], prism[5]);
  ray[5].set(prism[4], prism[5]);

  ray[6].set(prism[0], prism[3]);
  ray[7].set(prism[1], prism[4]);
  ray[8].set(prism[2], prism[5]);

  // Compute the intersection points of the plane and the prism
  xpoly3.erase(xpoly3.begin(), xpoly3.end());
  for(int i=0;i<n_edges;i++)
    {
      double x = P.intersect_ray(ray[i]);
      if ((x>=0)&&(x<=1))
      	{
	  // Intersection with ray occurs within the line segment defined
	  // by the edge, so add intersection point to our list 
	  xpoly3.push_back(ray[i].get_point(x));
	}
    }
  
  // Transform the triangle into 2D
  
  tpoly.resize(3);
  temp[0] = tri[0][0];temp[1]=tri[0][1];temp[2]=tri[0][2];
  P.map(temp, tpoly[0]);
  temp[0] = tri[1][0];temp[1]=tri[1][1];temp[2]=tri[1][2];
  P.map(temp, tpoly[1]); 
  temp[0] = tri[2][0];temp[1]=tri[2][1];temp[2]=tri[2][2];
  P.map(temp, tpoly[2]);

  // Transform the prism intersection points into 2D
  ppoly.resize(xpoly3.size());
  for(short i=0;i<ppoly.size();i++)
    {
      P.map(xpoly3[i], ppoly[i]);
    }


  //cerr << "Starting X test..." << endl;
  //cerr << "Tri : ";
  // for(int i =0;i<tpoly.size();i++) {cerr << "( " << tpoly[i][0] << " , " << tpoly[i][1] << ")";}
  // cerr << endl;
  // cerr << "Pri : ";
  // for(int i =0;i<ppoly.size();i++) {cerr << "( " << ppoly[i][0] << " , " << ppoly[i][1] << ")";}
  // cerr << endl;

   xpoly.resize(ppoly.size());
   
  // Radially sort the 2D prism polygon
   radial_sort2D(ppoly);
   radial_sort2D(tpoly);
   bool xfound = false;

   //FIX check that ppoly is tri or quad
   if( poly_in_convpoly2D(ppoly, tpoly))
     {//P inside Q
       xfound = true;
       xpoly2D.resize(ppoly.size());
       for(int i=0;i<ppoly.size();i++)
	 xpoly2D[i]=ppoly[i];
     }
   else if  (poly_in_convpoly2D(tpoly, ppoly))
     {//Q inside P
       xfound = true;
       xpoly2D.resize(tpoly.size());
       for(int i=0;i<tpoly.size();i++)
	 xpoly2D[i]=tpoly[i];
     } 

 // Find the 2D polygon of interection between the triangle 
 // and the trace of the prism 

 if (!xfound)
   {
     if (ppoly.size()>2)
       {
	 convex_poly_X( tpoly, ppoly, xpoly2D );	
       }
     else
       xpoly2D.erase(xpoly2D.begin(),xpoly2D.end());
   }

 //Remove duplicate points
 if (xpoly2D.size()>0)
   erase_consecutive_dups(xpoly2D);

 // Compute the area of the intersection
  if (xpoly2D.size() > 2)
    area = poly_area2D(xpoly2D);
  else
    area =0;
  
  xpoly.erase(xpoly.begin(),xpoly.end());
  xpoly.resize(xpoly2D.size());
  //  cerr << "Polygon size is " << xpoly2D.size() << endl;
  for(short i=0;i<xpoly2D.size();i++)
    {
      //cerr << "(" << xpoly2D[i][0] << "," << xpoly2D[i][1] << ")  ";
      P.unmap(xpoly[i], xpoly2D[i]);   
    }
  //cerr << endl;

  // Test 3D area = 2D area

  //FIX BUG where entire tri returned

  //FIX extra vertices returned sometimes? Maybe duplicates
  //floating point makes identification of dups hard.
  //Shouldn't affect area computation.

  return area;
}

//**********************************************************************/
// Function:  convex_poly_X
//
// Finds intersection of a pair of convex 2D polygons.
//
// Returns: 2D polygon (or empty set) 
//
// Prameters:


//***********************************************************************/

int    convex_poly_X( tPolygond & P, tPolygond & Q, tPolygond & I )
{
   int     m, n;           /* vertices in P and Q resp. */
   int     a, b;           /* indices on P and Q (resp.) */
   int     a1, b1;         /* a-1, b-1 (resp.) */
   tPointd A, B;           /* directed edges on P and Q (resp.) */
   int     cross;          /* sign of z-component of A x B */
   int     bHA, aHB;       /* b in H(A); a in H(b). */
   tPointd Origin(0,0);    /* (0,0) */
   tPointd p;              /* double point of intersection */
   tPointd q;              /* second point of intersection */
   tInFlag inflag;         /* {Pin, Qin, Unknown}: which inside */
   int     aa, ba;         /* # advances on a & b indices (after 1st inter.) */
   bool    FirstPoint;     /* Is this the first point? (used to initialize).*/
   tPointd p0;             /* The first point. */
   int     code;           /* SegSegInt return code. */ 
   bool    xfound;
   /* Initialize variables. */
   n = P.size();
   m = Q.size();
   a = 0; b = 0; aa = 0; ba = 0;
   inflag = Unknown; FirstPoint = true;
   I.erase(I.begin(),I.end());

   do {
      /* Computations of key variables. */
      a1 = (a + n - 1) % n;
      b1 = (b + m - 1) % m;
      Origin[0] =0;Origin[1]=0;
      A=P[a] - P[a1];
      B=Q[b] - Q[b1];
      //cerr << "If A & B intersect, update inflag..." << endl;
      cross = area_sign( Origin, A, B );
      aHB   = area_sign( Q[b1], Q[b], P[a] );
      bHA   = area_sign( P[a1], P[a], Q[b] );

      /* If A & B intersect, update inflag. */
      code = seg_seg_int( P[a1], P[a], Q[b1], Q[b], p, q );
      //printf("%%SegSegInt: code = %c\n", code );
      if ( code == '1' || code == 'v' ) {
         if ( inflag == Unknown && FirstPoint ) {
            aa = ba = 0;
            FirstPoint = false;
            p0[0] = p[0]; p0[1] = p[1];
	    I.push_back(p0);
         }
         inflag = in_out( p, inflag, aHB, bHA, I);
	 // printf("%%InOut sets inflag=%d\n", inflag);
      }

      /*-----Advance rules-----*/

      /* Special case: A & B overlap and oppositely oriented. */
      if ( ( code == 'e' ) && ((A*B) < 0) )
	{
	  shared_seg( p, q , I); 
	  return EXIT_SUCCESS;
	}

      /* Special case: A & B parallel and separated. */
      if ( (cross == 0) && ( aHB < 0) && ( bHA < 0 ) )
	{
	  return EXIT_SUCCESS;
	}

      /* Special case: A & B collinear. */
      else if ( (cross == 0) && ( aHB == 0) && ( bHA == 0 ) ) {
            /* Advance but do not output point. */
            if ( inflag == Pin )
	      b = advance( b, &ba, m, inflag == Qin, Q[b],I );
            else
	      a = advance( a, &aa, n, inflag == Pin, P[a], I );
         }

      /* Generic cases. */
      else if ( cross >= 0 ) {
         if ( bHA > 0)
	   a = advance( a, &aa, n, inflag == Pin, P[a], I );
         else
	   b = advance( b, &ba, m, inflag == Qin, Q[b], I );
      }
      else /* if ( cross < 0 ) */{
         if ( aHB > 0)
	   b = advance( b, &ba, m, inflag == Qin, Q[b], I );
         else
	   a = advance( a, &aa, n, inflag == Pin, P[a], I );
      }

   /* Quit when both adv. indices have cycled, or one has cycled twice. */
   } while ( ((aa < n) || (ba < m)) && (aa < 2*n) && (ba < 2*m) );

   if ( !FirstPoint ) /* If at least one point output, close up. */
     {
	    I.push_back(p0); //FIX
     }
   /* Deal with special cases: not implemented. */
   if ( inflag == Unknown) 
     {
       xfound = false;
      
       
       if (!xfound)
	 {
	   printf("%%The boundaries of P and Q do not cross.\n");
	   I.erase(I.begin(),I.end());
	 }
     }
   //FIX remove duplicates
   return EXIT_FAILURE;//FIX
}

/*---------------------------------------------------------------------
Prints out the double point of intersection, and toggles in/out flag.
---------------------------------------------------------------------*/
tInFlag in_out( tPointd & p, tInFlag inflag, int aHB, int bHA, tPolygond & I )
{
  I.push_back(p);
   /* Update inflag. */
   if      ( aHB > 0)
      return Pin;
   else if ( bHA > 0)
      return Qin;
   else    /* Keep status quo. */
      return inflag;
}
/*---------------------------------------------------------------------
   Advances and prints out an inside vertex if appropriate.
---------------------------------------------------------------------*/
int     advance( int a, int *aa, int n, bool inside, tPointd & v , tPolygond & I)
{
   if ( inside )
     {
       //printf("%5f    %5f    lineto\n", v[0], v[1] );
      I.push_back(v);
     }
   (*aa)++;
   return  (a+1) % n;
}

// /*
//    Reads in the coordinates of the vertices of a polygon from stdin,
//    puts them into P, and returns n, the number of vertices.
//    Formatting conventions: etc.
// */

int   ReadPoly( tPolygond & P )
{
  tPointd v;
   int   n = 0;
   int   nin;
   double x,y;
   scanf("%d", &nin);
   //printf("%%Polygon:\n");
   //printf("%%  i   x   y\n");
   P.resize(nin);
   while (n < nin)
     {
       cin >> x;
       cin >> y;
       v[0] = x;
       v[1] = y;
       P[n]=v;
       //cout << "%% " << n << "  " << P[n][0] << "   " << P[n][1] <<  endl;
      ++n;
   }

   //printf("%%n = %3d vertices read\n",n);
   putchar('\n');

   return   n;
}

/*
   Returns true iff c is strictly to the left of the directed
   line through a to b.
*/
bool    left( tPointd & a, tPointd & b, tPointd & c )
{
        return  area_sign( a, b, c ) > 0;
}

bool    left_on( tPointd & a, tPointd & b, tPointd & c )
{
        return  area_sign( a, b, c ) >= 0;
}

bool    collinear( tPointd & a, tPointd & b, tPointd & c )
{
        return  area_sign( a, b, c ) == 0;
}

int	area_sign( tPointd & a, tPointd & b, tPointd & c )
{
    double area2;

    area2 = ( b[0] - a[0] ) * ( c[1] - a[1] ) -
            ( c[0] - a[0] ) * ( b[1] - a[1] );

   
    if      ( area2 >  0.0 ) return  1;
    else if ( area2 <  0.0 ) return -1;
    else                     return  0;
}

/*---------------------------------------------------------------------
SegSegInt: Finds the point of intersection p between two closed
segments ab and cd.  Returns p and a char with the following meaning:
   'e': The segments collinearly overlap, sharing a point.
   'v': An endpoint (vertex) of one segment is on the other segment,
        but 'e' doesn't hold.
   '1': The segments intersect properly (i.e., they share a point and
        neither 'v' nor 'e' holds).
   '0': The segments do not intersect (i.e., they share no points).
Note that two collinear segments that share just one point, an endpoint
of each, returns 'e' rather than 'v' as one might expect.
---------------------------------------------------------------------*/
char	seg_seg_int( tPointd & a, tPointd & b, tPointd & c, 
		   tPointd & d, tPointd & p, tPointd & q )
{
   double  s, t;       /* The two parameters of the parametric eqns. */
   double num, denom;  /* Numerator and denoninator of equations. */
   char code = '?';    /* Return char characterizing intersection. */

   denom = a[0] * (double)( d[1] - c[1] ) +
           b[0] * (double)( c[1] - d[1] ) +
           d[0] * (double)( b[1] - a[1] ) +
           c[0] * (double)( a[1] - b[1] );

   /* If denom is zero, then segments are parallel: handle separately. */
   if (denom == 0.0)
      return  parallel_int(a, b, c, d, p, q);

   num =    a[0] * (double)( d[1] - c[1] ) +
            c[0] * (double)( a[1] - d[1] ) +
            d[0] * (double)( c[1] - a[1] );
   if ( (num == 0.0) || (num == denom) ) code = 'v';
   s = num / denom;

   num = -( a[0] * (double)( c[1] - b[1] ) +
            b[0] * (double)( a[1] - c[1] ) +
            c[0] * (double)( b[1] - a[1] ) );
   if ( (num == 0.0) || (num == denom) ) code = 'v';
   t = num / denom;
  
   if      ( (0.0 < s) && (s < 1.0) &&
             (0.0 < t) && (t < 1.0) )
     code = '1';
   else if ( (0.0 > s) || (s > 1.0) ||
             (0.0 > t) || (t > 1.0) )
     code = '0';

   p[0] = a[0] + s * ( b[0] - a[0] );
   p[1] = a[1] + s * ( b[1] - a[1] );

   return code;
}

char   parallel_int( tPointd & a, tPointd & b, tPointd & c, tPointd & d, tPointd & p, tPointd & q )
{

   if ( !collinear( a, b, c) )
      return '0';

   if ( between( a, b, c ) && between( a, b, d ) ) {
      p = c;
      q = d;
      return 'e';
   }
   if ( between( c, d, a ) && between( c, d, b ) ) {
      p = a;
      q = b;
      return 'e';
   }
   if ( between( a, b, c ) && between( c, d, b ) ) {
      p=c;
      q=b;
      return 'e';
   }
   if ( between( a, b, c ) && between( c, d, a ) ) {
      p=c;
      q=a;
      return 'e';
   }
   if ( between( a, b, d ) && between( c, d, b ) ) {
      p=d;
      q=b;
      return 'e';
   }
   if ( between( a, b, d ) && between( c, d, a ) ) {
      p=d;
      q=a;
      return 'e';
   }
   return '0';
}

/*---------------------------------------------------------------------
Returns true iff point c lies on the closed segement ab.
Assumes it is already known that abc are collinear.
---------------------------------------------------------------------*/
bool    between( tPointd & a, tPointd & b, tPointd & c )
{
   tPointd      ba, ca;

   /* If ab not vertical, check betweenness on x; else on y. */
   if ( a[0] != b[0] )
      return ((a[0] <= c[0]) && (c[0] <= b[0])) ||
             ((a[0] >= c[0]) && (c[0] >= b[0]));
   else
      return ((a[1] <= c[1]) && (c[1] <= b[1])) ||
             ((a[1] >= c[1]) && (c[1] >= b[1]));
}

void	shared_seg( tPointd & p, tPointd & q, tPolygond & I )
{
   I.push_back(p);
   I.push_back(q);
}
/********************************************************************/

#include "pgm.h"

//Compute forces on constant-strain triangles:
void CST_NL(const vector2d *coor,const connRec *lm,vector2d *R_net,
	    const vector2d *d,const double *c,
	    int numnp,int numel,
	    double *S11o,double *S22o,double *S12o)
{
  int n1,n2,n3,i;
  double S11,S22,S12,u1,u2,u3,v1,v2,v3,x21,y21,x31,y31,x32,y32;
  double aa,B1,B2,B3,B4,B5,B6,dudx,dvdy,dudy,dvdx;
  double E11,E22,E12;

  for (i=0;i<numel;i++) {
    n1=lm[i][0];
    n2=lm[i][1];
    n3=lm[i][2];
          u1 = d[n1].x;
          u2 = d[n2].x;
          u3 = d[n3].x;
          v1 = d[n1].y;
          v2 = d[n2].y;
          v3 = d[n3].y;

          x21 = coor[n2].x-coor[n1].x;
          y21 = coor[n2].y-coor[n1].y;
          x31 = coor[n3].x-coor[n1].x;
          y31 = coor[n3].y-coor[n1].y;
          x32 = coor[n3].x-coor[n2].x;
          y32 = coor[n3].y-coor[n2].y;

          aa = x21*y31-x31*y21;
          B1 = -y32/aa;
          B2 = x32/aa;
          B3 = y31/aa;
          B4 = -x31/aa;
          B5 = -y21/aa;
          B6 = x21/aa;

          dudx = B1*u1 + B3*u2 + B5*u3;
          dvdy = B2*v1 + B4*v2 + B6*v3;
          dudy = B2*u1 + B4*u2 + B6*u3;
          dvdx = B1*v1 + B3*v2 + B5*v3;
          E11 = dudx + 0.5*(dudx*dudx + dvdx*dvdx);
          E22 = dvdy + 0.5*(dvdy*dvdy + dudy*dudy);
          E12 = dudy + dvdx + dudx*dudy + dvdy*dvdx;

          // Calculate CST stresses
          S11 = E11*c[0] + E22*c[1];
          S22 = E11*c[1] + E22*c[2];
          S12 = E12*c[3];
          S11o[i]=S11;
          S22o[i]=S22;
          S12o[i]=S12;
	  
          // Calculate R-internal force vector
          R_net[n1] -= aa*0.5*vector2d(
               S11*B1*(1.0+dudx) +                 
               S22*B2*dudy +                        
               S12*(B2*(1.0+dudx) + B1*dudy)
	    ,
               S11*B1*dvdx +                        
               S22*B2*(1.0+dvdy) +                 
               S12*(B1*(1.0+dvdy)+B2*dvdx)
	    );
          R_net[n2] -= aa*0.5*vector2d(   
               S11*B3*(1.0+dudx) +                 
               S22*B4*dudy +                        
               S12*(B4*(1.0+dudx) + B3*dudy)
	    ,
	       S11*B3*dvdx +                        
               S22*B4*(1.0+dvdy) +                 
               S12*(B3*(1.0+dvdy)+B4*dvdx)
	    );
          R_net[n3] -= aa*0.5*vector2d(   
               S11*B5*(1.0+dudx) +                 
               S22*B6*dudy +                        
               S12*(B6*(1.0+dudx) + B5*dudy)
	    ,
               S11*B5*dvdx +                        
               S22*B6*(1.0+dvdy) +                 
               S12*(B5*(1.0+dvdy)+B6*dvdx)
	    ); 
  }
}

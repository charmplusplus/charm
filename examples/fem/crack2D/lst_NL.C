/**
 * Volumetric element physics for crack propagation code.
 *  converted from Fortran on 9/7/99 by Orion Sky Lawlor
 */
#include "crack.h"

//BtoR: sum the "B" spatial derivative array into the
// "R" partial internal force array.

static void BtoR(const Coord *B/*6-array*/,
    Coord *R/*6-array*/,Node **n/*6-array*/,
    Vol *v,const VolMaterial *vm,
    int pkCount,double iaa)

{
  double dudx,dvdy,dudy,dvdx;      //    ! partial derivatives of disp
  double E11,E22,E12;              //            ! strains
  int k;                         //Loop index

//C-----Calculate displacement gradient (H)
  dudx=dvdy=dudy=dvdx=0.0;
  for (k=0;k<6;k++) {
    dudx+=B[k].x*n[k]->disp.x *iaa;
    dvdy+=B[k].y*n[k]->disp.y *iaa;
    dudy+=B[k].y*n[k]->disp.x *iaa;
    dvdx+=B[k].x*n[k]->disp.y *iaa;
  }

//C-----Calculate Lagrangian strain (E)
  
  E11 = dudx + 0.5*(dudx*dudx + dvdx*dvdx);
  E22 = dvdy + 0.5*(dvdy*dvdy + dudy*dudy);
  E12 = dudy + dvdx + dudx*dudy + dvdy*dvdx;
  
//C-----Calculate the pkCount'th Piola-Kirchhoff stress (S)
  
  double s11c,s12c,s22c;
  s11c=v->s11l[pkCount] = E11*vm->c[0] + E22*vm->c[1];
  s22c=v->s22l[pkCount] = E11*vm->c[1] + E22*vm->c[2];
  s12c=v->s12l[pkCount] = E12*vm->c[3];
  
//Update R
  for (k=0;k<6;k++) {
    R[k].x+=s11c*B[k].x*(1.0+dudx)  +  s22c*B[k].y* dudy+
      s12c*(B[k].y*(1.0+dudx) + B[k].x*dudy);
    R[k].y+=s11c*B[k].x*dvdx        +  s22c*B[k].y*(1.0+dvdy)+
      s12c*(B[k].y*     dvdx  + B[k].x*(1.0+dvdy));
  }

}


void
lst_NL(MeshData *mesh)
{
  int idx;
  for(idx=0; idx<mesh->ne; idx++) {
    Vol *v = &(mesh->vols[idx]);
    Node *n[6];
    int k;
    for(k=0;k<6;k++)
      n[k] = &(mesh->nodes[v->conn[k]]);

    const double c5v3 = 1.66666666666667;
    const double c1v3 = 0.33333333333333;
    const double c2v3 = 0.66666666666667;
    const double c1v6 = 0.166666666666667;
  
    double x21,y21,x31,y31,x32,y32;         //! coor(1,n2)-coor(1,n1)
    double iaa;                       //! inverse of twice the element area
      
    Coord B[6];                           //! spacial derivatives
    Coord R[6];                           //! partial internal force
      
    //Nodes 0, 1, and 2 are the outer corners;
    //Nodes 4, 5, and 6 are on the midpoints of the sides.
      
    VolMaterial *vm = &(config.volm[v->material]); //Current material
      
    for (k=0;k<6;k++) {
        R[k].x=R[k].y=0.0;
    }
      
    x21 = n[1]->pos.x-n[0]->pos.x;
    y21 = n[1]->pos.y-n[0]->pos.y;
    x31 = n[2]->pos.x-n[0]->pos.x;
    y31 = n[2]->pos.y-n[0]->pos.y;
    x32 = n[2]->pos.x-n[1]->pos.x;
    y32 = n[2]->pos.y-n[1]->pos.y;
    iaa = 1.0/(x21*y31-x31*y21);
      
    //Perform the three stages of Piola-Kirchoff stress
    //First stage:
    B[0].x = -y32*c5v3;             //                !   B1 = aa*dN1/dx
    B[0].y = x32*c5v3;              //                !   B2 = aa*dN1/dy
    B[1].x = -y31*c1v3;             //                !   B3 = aa*dN2/dx
    B[1].y = x31*c1v3;              //                !   B4 = aa*dN2/dy
    B[2].x = y21*c1v3;              //                !   B5 = aa*dN3/dx
    B[2].y = -x21*c1v3;             //                !   B6 = aa*dN3/dy
    B[3].x = (4.0*y31 - y32)*c2v3;  //                !   B7 = aa*dN4/dx
    B[3].y = (x32 - 4.0*x31)*c2v3;  //                !   B8 = aa*dN4/dy
    B[4].x = (y31 - y21)*c2v3;      //                !   B9 = aa*dN5/dx
    B[4].y = (x21 - x31)*c2v3;      //                !   B10 = aa*dN5/dy
    B[5].x = -(y32 + 4.0*y21)*c2v3; //                !   B11 = aa*dN6/dx
    B[5].y = (x32 + 4.0*x21)*c2v3;  //                !   B12 = aa*dN6/dy
          
    BtoR(B,R,n,v,vm,0,iaa);
          
    //Second stage:
    B[0].x = y32*c1v3;               //   !   B1 = aa*dN1/dx
    B[0].y = -x32*c1v3;              //   !   B2 = aa*dN1/dy
    B[1].x = y31*c5v3;               //   !   B3 = aa*dN2/dx
    B[1].y = -x31*c5v3;              //   !   B4 = aa*dN2/dy
    B[2].x = y21*c1v3;               //   !   B5 = aa*dN3/dx
    B[2].y = -x21*c1v3;              //   !   B6 = aa*dN3/dy
    B[3].x = (y31 - 4.0*y32)*c2v3;   //   !   B7 = aa*dN4/dx
    B[3].y = (4.0*x32 - x31)*c2v3;   //   !   B8 = aa*dN4/dy
    B[4].x = (y31 - 4.0*y21)*c2v3;   //   !   B9 = aa*dN5/dx
    B[4].y = (4.0*x21 - x31)*c2v3;   //   !   B10 = aa*dN5/dy
    B[5].x = -(y32 + y21)*c2v3;      //   !   B11 = aa*dN6/dx
    B[5].y = (x32 + x21)*c2v3;       //   !   B12 = aa*dN6/dy
    
    BtoR(B,R,n,v,vm,1,iaa);
    
    //Third stage:
    B[0].x = y32*c1v3;               //  !   B1 = aa*dN1/dx
    B[0].y = -x32*c1v3;              //  !   B2 = aa*dN1/dy
    B[1].x = -y31*c1v3;              //  !   B3 = aa*dN2/dx
    B[1].y = x31*c1v3;               //  !   B4 = aa*dN2/dy
    B[2].x = -y21*c5v3;              //  !   B5 = aa*dN3/dx
    B[2].y = x21*c5v3;               //  !   B6 = aa*dN3/dy
    B[3].x = (y31 - y32)*c2v3;       //  !   B7 = aa*dN4/dx
    B[3].y = (x32 - x31)*c2v3;       //  !   B8 = aa*dN4/dy
    B[4].x = (4.0*y31 - y21)*c2v3;   //  !   B9 = aa*dN5/dx
    B[4].y = (x21 - 4.0*x31)*c2v3;   //  !   B10 = aa*dN5/dy
    B[5].x = -(4.0*y32 + y21)*c2v3;  //  !   B11 = aa*dN6/dx
    B[5].y = (4.0*x32 + x21)*c2v3;   //  !   B12 = aa*dN6/dy
    
    BtoR(B,R,n,v,vm,2,iaa);
    
    //Now we've calculated R.  Update each node`s Rin with the new values.
    for (k=0;k<6;k++) {
      n[k]->Rin.x+=c1v6*R[k].x;
      n[k]->Rin.y+=c1v6*R[k].y;
    }
  }
}

/**
 * Cohesive element physics for crack propagation code.
 */
#include "crack.h"
#include <math.h>


//Update cohesive element traction itNo given Dn, Dt
static void 
updateTraction(double Dn, double Dt,Coh *coh,CohMaterial *m,int itNo)
{
  double Tn, Tt;
  double c=coh->sidel[1],s=coh->sidel[2];
  double x = 1.0 - sqrt(Dn*Dn+Dt*Dt);
  
  if (x <= 0.0) {
    coh->Sthresh[itNo] = 0.0;
  } else if (x <= coh->Sthresh[itNo])
    coh->Sthresh[itNo] = x;
  
  if (Dn > 0.0)
    Tn = coh->Sthresh[itNo]/(1.0-coh->Sthresh[itNo])*m->sigmax*Dn;
  else
    Tn = m->Sinit/(1.0-m->Sinit)*m->sigmax*Dn;
  
  Tt = coh->Sthresh[itNo]/(1.0-coh->Sthresh[itNo])*m->taumax*Dt;
  coh->T[itNo].x = c*Tt - s*Tn;
  coh->T[itNo].y = s*Tt + c*Tn;
}

void
lst_coh2(MeshData *mesh) 
{
  int idx;
  for(idx=0; idx<mesh->nc; idx++) {
    Coh *coh = &(mesh->cohs[idx]);
    Node *n[6];
    int k;
    for(k=0;k<6;k++)
      n[k] = &(mesh->nodes[coh->conn[k]]);

    // local variables:
    double deltn, deltt;  // the char length of current element
    double length, c, s;
    double Dx1, Dx2, Dx3;
    double Dy1, Dy2, Dy3;
    double Dn1, Dn2, Dn3;
    double Dt1, Dt2, Dt3;
    double Dn, Dt;
    double x;
    double Rx1,Rx2,Rx3;  // cohesive forces at nodes 1,2,5
    double Ry1,Ry2,Ry3;  // cohesive forces at nodes 1,2,5
  
    // g1,g3: gauss quadrature points
    // w1,w2,w3: weights; w1=w2, w3=w4
    
    CohMaterial *m = &(config.cohm[coh->material]);
      
    deltn = (double)1.0 / m->deltan;
    deltt = (double)1.0 / m->deltat;
    length = coh->sidel[0] * (double)0.5;
    c = coh->sidel[1];
    s = coh->sidel[2];
    
    Dx1 = n[3]->disp.x - n[0]->disp.x;
    Dy1 = n[3]->disp.y - n[0]->disp.y;
    Dt1 =  (c*Dx1 + s*Dy1)*deltt;
    Dn1 = (-s*Dx1 + c*Dy1)*deltn;
    Dx2 = n[2]->disp.x - n[1]->disp.x;
    Dy2 = n[2]->disp.y - n[1]->disp.y;
    Dt2 =  (c*Dx2 + s*Dy2)*deltt;
    Dn2 = (-s*Dx2 + c*Dy2)*deltn;
    Dx3 = n[5]->disp.x - n[4]->disp.x;
    Dy3 = n[5]->disp.y - n[4]->disp.y;
    Dt3 =  (c*Dx3 + s*Dy3)*deltt;
    Dn3 = (-s*Dx3 + c*Dy3)*deltn;
      
    // gauss point 1
    Dt = g1*g1*0.5*(Dt1+Dt2-2.0*Dt3)+g1*0.5*(Dt2-Dt1)+Dt3;
    Dn = g1*g1*0.5*(Dn1+Dn2-2.0*Dn3)+g1*0.5*(Dn2-Dn1)+Dn3;
    updateTraction(Dn,Dt,coh,m,0);
      
    // gauss point 2
    updateTraction(Dn3,Dt3,coh,m,1);
  
    // gauss point 3
    Dt = g3*g3*0.5*(Dt1+Dt2-2.0*Dt3)+g3*0.5*(Dt2-Dt1)+Dt3;
    Dn = g3*g3*0.5*(Dn1+Dn2-2.0*Dn3)+g3*0.5*(Dn2-Dn1)+Dn3;
    updateTraction(Dn,Dt,coh,m,2);
  
    // cohesive forces
    x = length*w1*g1*0.5;
    Rx1 = (coh->T[0].x*(g1-1.0)+coh->T[2].x*(g1+1.0))*x;
    Ry1 = (coh->T[0].y*(g1-1.0)+coh->T[2].y*(g1+1.0))*x;
    Rx2 = (coh->T[0].x*(g1+1.0)+coh->T[2].x*(g1-1.0))*x;
    Ry2 = (coh->T[0].y*(g1+1.0)+coh->T[2].y*(g1-1.0))*x;
    Rx3 = ((coh->T[0].x+coh->T[2].x)*w1*(1.0-g1*g1)+coh->T[1].x*w2)*length;
    Ry3 = ((coh->T[0].y+coh->T[2].y)*w1*(1.0-g1*g1)+coh->T[1].y*w2)*length;
    n[0]->Rco.x += Rx1;
    n[0]->Rco.y += Ry1;
    n[3]->Rco.x -= Rx1;
    n[3]->Rco.y -= Ry1;
    n[1]->Rco.x += Rx2;
    n[1]->Rco.y += Ry2;
    n[2]->Rco.x -= Rx2;
    n[2]->Rco.y -= Ry2;
    n[4]->Rco.x += Rx3;
    n[4]->Rco.y += Ry3;
    n[5]->Rco.x -= Rx3;
    n[5]->Rco.y -= Ry3;
  }
}

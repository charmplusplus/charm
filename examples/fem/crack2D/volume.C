#include "crack.h"
#include <math.h>

void 
vol_elem(GlobalData *gd)
{
  double x, x1, x2, x3;

  //Compute the elastic stiffness constants for each material type
  int matNo;
  for (matNo=0;matNo<gd->numMatVol;matNo++)
  {
    VolMaterial *vm=&(gd->volm[matNo]);
    switch (gd->nplane)
    {
      case 1://Orthotropic plane strain
        double sT,cT,xnu21;
        cT = cos(vm->theta*1.74532925199e-2);
        sT = sin(vm->theta*1.74532925199e-2);
        xnu21 = vm->xnu12*vm->e2/vm->e1;
        x = 1.0 - vm->xnu23*vm->xnu23 - 
          2.0*vm->xnu12*xnu21*(1.0 + vm->xnu23);
        x1 = vm->e1*(1.0-vm->xnu23*vm->xnu23) / x;
        x2 = xnu21*vm->e1*(1.0+vm->xnu23) / x;
        x3 = vm->e2*(vm->xnu23+vm->xnu12*xnu21) / x;
        vm->c[2] = vm->e2*(1.0-vm->xnu12*xnu21) / x;
        vm->c[0] = x1*cT*cT*cT*cT + 2.0*(x2+2.0*vm->g12)*cT*cT*sT*sT +
          vm->c[2]*sT*sT*sT*sT;
        vm->c[1] = x2*cT*cT + x3*sT*sT;
        vm->c[3] = vm->g12*cT*cT + vm->g23*sT*sT;
        break;
      case 0: //Plane stress (isotropic)
        vm->c[0] = vm->e1 / (1.0 - vm->xnu12*vm->xnu12);
        vm->c[1] = vm->e1*vm->xnu12 / (1.0 - vm->xnu12*vm->xnu12);
        vm->c[2] = vm->c[0];
        vm->c[3] = vm->e1/ (2.0 * (1.0 + vm->xnu12));
        break;
      case 2: //Axisymmetric (isotropic)
        vm->c[0] = vm->e1 * (1.0 - vm->xnu12) / ((1.0 + vm->xnu12)*
                                                 (1.0 - 2.0*vm->xnu12));
        vm->c[1] = vm->e1 * vm->xnu12 / ((1.0 + vm->xnu12)*
                                         (1.0 - 2.0*vm->xnu12));
        vm->c[2] = vm->e1 / (2.0*(1.0 + vm->xnu12));
        break;
      default:
        CkAbort("Unknown planar analysis type passed to vol_elem!\n");
    }
  }

  //Update the node-by-node mass of each element
  int volNo;
  for (volNo=gd->svol;volNo<gd->ne;volNo++)
  {
    Node *n[6];              //Pointers to each of the triangle's nodes
    int k;                  //Loop index
    for (k=0;k<6;k++)
      n[k]=&(gd->nodes[gd->conn[volNo*6+k]]);
    //Compute the mass of this element
    double area=((n[1]->pos.x-n[0]->pos.x)*(n[2]->pos.y-n[0]->pos.y)-
                 (n[2]->pos.x-n[0]->pos.x)*(n[1]->pos.y-n[0]->pos.y));
    double mass=gd->volm[gd->elements[volNo].material].rho*area/114.0;
    //Divide the mass among the element's nodes
    for (k=0;k<3;k++) {
      n[k]->xM.x+=mass*3.0;
      n[k]->xM.y+=mass*3.0;
    }
    for (k=3;k<6;k++) {
      n[k]->xM.x+=mass*16.0;
      n[k]->xM.y+=mass*16.0;
    }
  }  
}

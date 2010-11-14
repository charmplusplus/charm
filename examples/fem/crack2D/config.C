/**
 * Read configuration file for crack propagation code.
 */

#include "converse.h"

#include <fstream>
using namespace std;

#include <stddef.h>
#include "crack.h"

/// Globally accessible (write-once) configuration data.
ConfigurationData config;

#define MAXLINE 1024
// Silly "skip n lines" macro.  Should actually only skip 
//  comment lines, but this will have to do...
#define cl(fd, buffer, n) do {\
                               for(int _i=0; _i<(n); _i++) \
                                 fd.getline(buffer, MAXLINE);\
                             } while(0)

/// This routine reads the config data from the config file.
void readConfig(const char *configFile,const char *meshFile)
{
  ifstream fg; // global parameter file
  ifstream fm; // mesh file, for reading loading data
  ConfigurationData *cfg=&config; //Always write to the global version
  char buf[MAXLINE];
  int i, itmp;
  double dtmp;
  
  fg.open(configFile);
  fm.open(meshFile);
  if (!fg || !fm)
  {
    crack_abort("Cannot open configuration file for reading.\n");
    return;
  }
  cl(fg,buf,3); // ignore first three lines
  fg >> cfg->nTime >> cfg->steps >> cfg->tsintFull 
     >> cfg->tsintEnergy >> cfg->restart;
  cl(fg, buf, 2);
  fg >> cfg->nplane >> cfg->ncoh >> cfg->lin;
  cl(fg, buf, 2);
  
  //read volumetric material properties
  fg >> cfg->numMatVol;
  cfg->volm = new VolMaterial[cfg->numMatVol];
  VolMaterial *v;
  cl(fg, buf, 2);
  for (i=0; i<cfg->numMatVol; i++)
  {
    v = &(cfg->volm[i]);
    fg >> v->e1 >> v->e2 >> v->g12 >> v->g23;
    fg >> v->xnu12 >> v->xnu23 >> v->rho;
    fg >> v->alpha1 >> v->alpha2 >> v->theta;
    cl(fg, buf, 2);
  }
  
  //Compute the elastic stiffness constants for each material type
  for (int matNo=0;matNo<cfg->numMatVol;matNo++)
  {
    double x, x1, x2, x3;
    VolMaterial *vm=&(cfg->volm[matNo]);
    switch (cfg->nplane)
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
        crack_abort("Unknown planar analysis type in config file");
    }
  }

  //read cohesive material properties
  fg >> cfg->numMatCoh;
  cfg->cohm = new CohMaterial[cfg->numMatCoh];
  CohMaterial *c;
  cl(fg, buf, 2);
  for (i=0; i<cfg->numMatCoh; i++)
  {
    c = &(cfg->cohm[i]);
    fg >> c->deltan >> c->deltat >> c->sigmax
       >> c->taumax >> c->mu;
    if (cfg->ncoh)
      fg >> c->Sinit;
    cl(fg, buf, 2);
  }
  
  //read impact data
  fg >> cfg->imp >> cfg->voImp >> cfg->massImp >> cfg->radiusImp;
  cl(fg, buf, 2);
  fg >> cfg->eImp >> cfg->xnuImp;
  cl(fg, buf, 2);
  cfg->voImp = 0; cfg->massImp = 1.0; cfg->radiusImp = 0; 
  cfg->eImp = 1.0; cfg->xnuImp = 0.3;
  
  //read (& ignore) thermal load
  fg >> dtmp;
  fg.close();
  
  //read proportional ramp-up for boundary conditions
  fm >> itmp >> itmp >> cfg->delta >> cfg->numProp;
  cfg->delta /= (double) cfg->steps;
  cfg->delta2 = cfg->delta*cfg->delta*0.5;
  cfg->ts_proportion = new int[cfg->numProp];
  cfg->proportion = new double[cfg->numProp];
  for (i=0; i< cfg->numProp; i++) {
    fm >> cfg->ts_proportion[i] >> cfg->proportion[i];
  }
  fm.close();
}




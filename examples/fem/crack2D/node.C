/**
 * Node manipulation for crack propagation code.
 */
#include <stddef.h>
#include "crack.h"

void nodeSetup(NodeSlope *sl) {
  sl->slope=sl->prop=0;
}

void nodeBeginStep(MeshData *mesh)
{
  for(int idx=0; idx<mesh->nn; idx++) {
    Node *np = &(mesh->nodes[idx]);
    // Zero out node accumulators, update node positions
    np->Rco.y = np->Rco.x = np->Rin.y =  np->Rin.x = 0.0;
    np->disp.x += config.delta2*np->accel.x + config.delta*np->vel.x;
    np->disp.y += config.delta2*np->accel.y + config.delta*np->vel.y;
  }
}

/**
 Return the fraction of the boundary conditions (prop) and time-rate of
 change of boundary conditions (slope) to apply during this 0-based timestep.  
 Applies linear interpolation to the ts_proportion and proportion arrays, above.
*/
void ConfigurationData::getBoundaryConditionScale(int timestep,double *prop,double *slope) const
{
    timestep--; /* to 0-based */
    /* Clamp out-of-bounds timesteps */
    if (timestep<=ts_proportion[0]) {
      *prop=proportion[0]; *slope=0; return;
    }
    if (timestep>=ts_proportion[numProp-1]) {
      *prop=proportion[numProp-1]; *slope=0; return;
    }
    /* Otherwise linearly interpolate between proportion "cur" and "cur+1" */
    int cur=0; /* index into ts_proportion and proportion arrays */
    while (timestep>=ts_proportion[cur+1]) cur++;
    /* assert: 0<=cur && cur<numProp 
      and  ts_proportion[cur]<=timestep && timestep<ts_proportion[cur+1] */
    
    *prop = proportion[cur];
    double timeSlope=1.0/(ts_proportion[cur+1]-ts_proportion[cur]);
    *slope = 1/delta * (proportion[cur+1]-proportion[cur])*timeSlope;
    *prop = proportion[cur]+(double)(timestep-ts_proportion[cur])*(*slope)*delta;
}


void nodeFinishStep(MeshData *mesh, NodeSlope *sl,int tstep)
{
  double prop,slope;
  config.getBoundaryConditionScale(tstep,&prop,&slope);

  // Update each node:
  for(int idx=0; idx<mesh->nn; idx++) {
    Node *np = &(mesh->nodes[idx]);
    if(!np->isbnd) { //Apply normal timestep to this node:
      double aX, aY;
      aX = (np->Rco.x-np->Rin.x)*np->xM.x;
      aY = (np->Rco.y-np->Rin.y)*np->xM.y;
      np->vel.x += (config.delta*(np->accel.x+aX)*(double) 0.5);
      np->vel.y += (config.delta*(np->accel.y+aY)*(double) 0.5);
      np->accel.x = aX;
      np->accel.y = aY;
    } else { //Apply boundary conditions to this node:
      double acc;
      if (!(np->id1)) {
        np->vel.x = (np->r.x)*prop;
        np->accel.x = (np->r.x)*slope;
      } else {
        acc = (np->r.x*prop+ np->Rco.x - np->Rin.x)*np->xM.x;
        np->vel.x += (config.delta*(np->accel.x+acc)*0.5);
        np->accel.x = acc;
      }
      if (!(np->id2)) {
        np->vel.y = np->r.y*prop;
        np->accel.y = np->r.y*slope;
      } else {
        acc = (np->r.y*prop+ np->Rco.y - np->Rin.y)*np->xM.y;
        np->vel.y = (np->vel.y + config.delta*(np->accel.y+acc)*0.5);
        np->accel.y = acc;
      }
    }
  }
}



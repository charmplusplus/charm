/**
 * Node manipulation for crack propagation code.
 */
#include <fstream.h>
#include <stddef.h>
#include "crack.h"

void nodeSetup(NodeSlope *sl) {
  sl->kk=-1;
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

void nodeFinishStep(MeshData *mesh, NodeSlope *sl,int tstep)
{
  // Slowly ramp up boundary conditions:
  if (config.ts_proportion[sl->kk+1] == tstep)
  {
      sl->kk++;
      sl->prop = config.proportion[sl->kk];
      sl->slope = (config.proportion[sl->kk+1]-sl->prop)/config.delta;
      sl->slope /= (double) (config.ts_proportion[sl->kk+1]- config.ts_proportion[sl->kk]);
  }
  else
  {
      sl->prop = (double)(tstep - config.ts_proportion[sl->kk])*
                    sl->slope*config.delta+config.proportion[sl->kk];
  }
  double slope=sl->slope;
  double prop=sl->prop;

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



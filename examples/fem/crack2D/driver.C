#include <fstream.h>
#include "crack.h"

extern "C" void
init(void)
{
}

static void
initNodes(GlobalData *gd)
{
  int idx;
  for(idx=0; idx<gd->nn; idx++) {
    Node *np = &(gd->nodes[idx]);
    // Zero out node accumulators, update node positions
    np->Rco.y = np->Rco.x = np->Rin.y =  np->Rin.x = 0.0;
    np->disp.x += gd->delta2*np->accel.x + gd->delta*np->vel.x;
    np->disp.y += gd->delta2*np->accel.y + gd->delta*np->vel.y;
  }
}

static void 
updateNodes(GlobalData *gd, double prop, double slope)
{
  for(int idx=0; idx<gd->nn; idx++) {
    Node *np = &(gd->nodes[idx]);
    if(!np->isbnd) {
      double aX, aY;
      aX = (np->Rco.x-np->Rin.x)*np->xM.x;
      aY = (np->Rco.y-np->Rin.y)*np->xM.y;
      np->vel.x += (gd->delta*(np->accel.x+aX)*(double) 0.5);
      np->vel.y += (gd->delta*(np->accel.y+aY)*(double) 0.5);
      np->accel.x = aX;
      np->accel.y = aY;
    } else {
      double acc;
      if (!(np->id1)) {
        np->vel.x = (np->r.x)*prop;
        np->accel.x = (np->r.x)*slope;
      } else {
        acc = (np->r.x*prop+ np->Rco.x - np->Rin.x)*np->xM.x;
        np->vel.x += (gd->delta*(np->accel.x+acc)*0.5);
        np->accel.x = acc;
      }
      if (!(np->id2)) {
        np->vel.y = np->r.y*prop;
        np->accel.y = np->r.y*slope;
      } else {
        acc = (np->r.y*prop+ np->Rco.y - np->Rin.y)*np->xM.y;
        np->vel.y = (np->vel.y + gd->delta*(np->accel.y+acc)*0.5);
        np->accel.y = acc;
      }
    }
  }
}

extern "C" void
driver(int nn, int *nnums, int ne, int *enums, int npere, int *conn)
{
  int myid = FEM_My_Partition();
  GlobalData *gd = new GlobalData;
  Node *nodes = new Node[nn];
  Element *elements = new Element[ne];
  gd->nn = nn;
  gd->nnums = nnums;
  gd->ne = ne;
  gd->enums = enums;
  gd->npere = npere;
  gd->conn = conn;
  gd->nodes = nodes;
  gd->elements = elements;
  readFile(gd);
  int i;
  int kk = -1;
  double prop, slope;
  for(i=0;i<gd->nTime;i++)
  {
    CkPrintf("[%d] Beginning iteration %d\n", myid, i);
    if (gd->ts_proportion[kk+1] == i)
    {
      kk++;
      prop = gd->proportion[kk];
      slope = (gd->proportion[kk+1]-prop)/gd->delta;
      slope /= (double) (gd->ts_proportion[kk+1]- gd->ts_proportion[kk]);
    }
    else
    {
      prop = (double)(i - gd->ts_proportion[kk])*
                    slope*gd->delta+gd->proportion[kk];
    }
    initNodes(gd);
    lst_NL(gd);
    lst_coh2(gd);
    updateNodes(gd, prop, slope);
    // FIXME: fem_update here.
  }
}

extern "C" void
finalize()
{
}

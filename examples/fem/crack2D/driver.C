#include <fstream.h>
#include <stddef.h>
#include "crack.h"

extern "C" void
init(void)
{
}

static int
mypksz(GlobalData *gd)
{
  int size = sizeof(GlobalData);
  size += gd->numProp*sizeof(int); // int *ts_proportion
  size += gd->numProp*sizeof(double); // double *proportion
  size += gd->numMatVol*sizeof(VolMaterial); // VolMaterial* volm
  size += gd->numMatCoh*sizeof(CohMaterial); // CohMaterial* cohm
  size += gd->nn*sizeof(Node); // Node *nodes
  size += gd->ne*sizeof(Element); // Element *elements
  return size;
}

static void
mypk(GlobalData *gd, void *buffer)
{
  char *buf = (char *) buffer;
  memcpy((void*)buf, (void*) gd, sizeof(GlobalData)); buf += sizeof(GlobalData);
  memcpy((void*)buf, (void*) gd->ts_proportion, gd->numProp*sizeof(int));
  buf += gd->numProp*sizeof(int);
  memcpy((void*)buf, (void*) gd->proportion, gd->numProp*sizeof(double));
  buf += gd->numProp*sizeof(double);
  memcpy((void*)buf, (void*) gd->volm, gd->numMatVol*sizeof(VolMaterial));
  buf += gd->numMatVol*sizeof(VolMaterial);
  memcpy((void*)buf, (void*) gd->cohm, gd->numMatCoh*sizeof(CohMaterial));
  buf += gd->numMatCoh*sizeof(CohMaterial);
  memcpy((void*)buf, (void*) gd->nodes, gd->nn*sizeof(Node));
  buf += gd->nn*sizeof(Node);
  memcpy((void*)buf, (void*) gd->elements, gd->ne*sizeof(Element));
  buf += gd->ne*sizeof(Element);
  delete[] gd->ts_proportion;
  delete[] gd->proportion;
  delete[] gd->volm;
  delete[] gd->cohm;
  delete[] gd->nodes;
  delete[] gd->elements;
}

static GlobalData *
myupk(void *buffer)
{
  char *buf = (char *) buffer;
  GlobalData *gd = new GlobalData;
  memcpy((void*) gd, (void*)buf, sizeof(GlobalData)); buf += sizeof(GlobalData);
  gd->ts_proportion = new int[gd->numProp];
  gd->proportion = new double[gd->numProp];
  gd->volm = new VolMaterial[gd->numMatVol];
  gd->cohm = new CohMaterial[gd->numMatCoh];
  gd->nodes = new Node[gd->nn];
  gd->elements = new Element[gd->ne];
  memcpy((void*) gd->ts_proportion, (void*)buf, gd->numProp*sizeof(int));
  buf += gd->numProp*sizeof(int);
  memcpy((void*) gd->proportion, (void*)buf, gd->numProp*sizeof(double));
  buf += gd->numProp*sizeof(double);
  memcpy((void*) gd->volm, (void*)buf, gd->numMatVol*sizeof(VolMaterial));
  buf += gd->numMatVol*sizeof(VolMaterial);
  memcpy((void*) gd->cohm, (void*)buf, gd->numMatCoh*sizeof(CohMaterial));
  buf += gd->numMatCoh*sizeof(CohMaterial);
  memcpy((void*) gd->nodes, (void*)buf, gd->nn*sizeof(Node));
  buf += gd->nn*sizeof(Node);
  memcpy((void*) gd->elements, (void*)buf, gd->ne*sizeof(Element));
  buf += gd->ne*sizeof(Element);

  return gd;
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

extern "C" double CmiCpuTimer(void);
static void
_DELAY_(int microsecs)
{
  double upto = CmiCpuTimer() + 1.e-6 * microsecs;
  while(upto > CmiCpuTimer());
}

extern "C" void
driver(int nn, int *nnums, int ne, int *enums, int npere, int *conn)
{
  int myid = FEM_My_Partition();
  // CkPrintf("[%d] starting driver\n", myid);
  GlobalData *gd = new GlobalData;
  FEM_Register((void*)gd, (FEM_Packsize_Fn)mypksz, (FEM_Pack_Fn)mypk,
               (FEM_Unpack_Fn)myupk);
  Node *nodes = new Node[nn];
  Element *elements = new Element[ne];
  gd->myid = myid;
  gd->nn = nn;
  gd->nnums = nnums;
  gd->ne = ne;
  gd->enums = enums;
  gd->npere = npere;
  gd->conn = conn;
  gd->nodes = nodes;
  gd->elements = elements;
  readFile(gd);
  int massfield = FEM_Create_Field(FEM_DOUBLE, 2, offsetof(Node, xM), 
                                   sizeof(Node));
  int rfield = FEM_Create_Field(FEM_DOUBLE, 4, offsetof(Node, Rin), 
                                sizeof(Node));
  FEM_Update_Field(massfield, gd->nodes);
  int i;
  int kk = -1;
  double prop, slope;
  double stime, etime;
  int phase = 0;
  for(i=0;i<gd->nTime;i++)
  {
    stime = CkWallTimer();
    // CkPrintf("[%d] iteration %d at %lf secs\n", gd->myid, i, CkTimer());
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
    if(myid==3 && i>20)
    {
      int biter = (i < 30 ) ? i : 30;
      _DELAY_((biter-20)*5000);
    }
    updateNodes(gd, prop, slope);
    FEM_Update_Field(rfield, gd->nodes);
    if(i%20==19)
    {
      FEM_Migrate();
      gd = (GlobalData*) FEM_Get_Userdata();
      gd->nnums = FEM_Get_Node_Nums();
      gd->enums = FEM_Get_Elem_Nums();
      gd->conn = FEM_Get_Conn();
    }
    etime = CkWallTimer();
    if(gd->myid == 0)
      CkPrintf("Iter=%d\tTime=%lf seconds\n", i, (etime-stime));
  }
  // CkPrintf("[%d] exiting driver\n", myid);
}

extern "C" void
finalize()
{
}

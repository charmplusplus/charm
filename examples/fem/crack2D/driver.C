#include <fstream.h>
#include <stddef.h>
#include "crack.h"

extern "C" void
init(void)
{
}

static GlobalData *
mypup(pup_er p, GlobalData *gd)
{
  if(pup_isUnpacking(p))
    gd = new GlobalData;
  pup_bytes(p, (void*)gd, sizeof(GlobalData));
  if(pup_isUnpacking(p))
  {
    gd->ts_proportion = new int[gd->numProp];
    gd->proportion = new double[gd->numProp];
    gd->volm = new VolMaterial[gd->numMatVol];
    gd->cohm = new CohMaterial[gd->numMatCoh];
    gd->itimes = new double[gd->nTime];
    gd->nodes = new Node[gd->nn];
    gd->elements = new Element[gd->ne];
  }
  pup_ints(p, gd->ts_proportion, gd->numProp);
  pup_doubles(p, gd->proportion, gd->numProp);
  pup_bytes(p, (void*)gd->volm, gd->numMatVol*sizeof(VolMaterial));
  pup_bytes(p, (void*) gd->cohm, gd->numMatCoh*sizeof(CohMaterial));
  pup_doubles(p, gd->itimes, gd->nTime);
  pup_bytes(p, (void*) gd->nodes, gd->nn*sizeof(Node));
  pup_bytes(p, (void*) gd->elements, gd->ne*sizeof(Element));
  if(pup_isPacking(p))
  {
    delete[] gd->ts_proportion;
    delete[] gd->proportion;
    delete[] gd->volm;
    delete[] gd->cohm;
    delete[] gd->itimes;
    delete[] gd->nodes;
    delete[] gd->elements;
    delete gd;
    gd = 0;
  }
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
  int numparts = FEM_Num_Partitions();
  // CkPrintf("[%d] starting driver\n", myid);
  GlobalData *gd = new GlobalData;
  int uidx;
  uidx = FEM_Register((void*)gd, (FEM_PupFn)mypup);
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
  gd->itimes = new double[gd->nTime];
  int i;
  for(i=0;i<gd->nTime;i++)
    gd->itimes[i] = 0.0;
  // CkPrintf("[%d] read file\n", myid);
  int massfield = FEM_Create_Field(FEM_DOUBLE, 2, offsetof(Node, xM), 
                                   sizeof(Node));
  int rfield = FEM_Create_Field(FEM_DOUBLE, 4, offsetof(Node, Rin), 
                                sizeof(Node));
  int tfield = FEM_Create_Field(FEM_DOUBLE, 1, 0, 0);
  FEM_Update_Field(massfield, gd->nodes);
  int kk = -1;
  double prop, slope;
  //double stime, etime;
  int phase = 0;
  //stime = CkWallTimer();
  for(i=0;i<gd->nTime;i++)
  {
    gd->itimes[i] = CkWallTimer();
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
    if(myid==79 && i>35)
    {
      int biter = (i < 40 ) ? i : 40;
      _DELAY_((biter-35)*19000);
    }
    updateNodes(gd, prop, slope);
    FEM_Update_Field(rfield, gd->nodes);
    if(i%25==24)
    {
      // if(gd->myid == 0)
      // {
        // etime = CkWallTimer();
        // CkPrintf("Phase=%d\tTime=%lf seconds\n", i, (etime-stime));
        phase++;
      // }
      FEM_Migrate();
      gd = (GlobalData*) FEM_Get_Userdata(uidx);
      gd->nnums = FEM_Get_Node_Nums();
      gd->enums = FEM_Get_Elem_Nums();
      gd->conn = FEM_Get_Conn();
      //stime = CkWallTimer();
    }
    gd->itimes[i] = CkWallTimer()-(gd->itimes[i]);
    FEM_Reduce(tfield, &gd->itimes[i], &gd->itimes[i], FEM_SUM);
  }
  if(gd->myid==0)
  {
    for(i=0;i<gd->nTime;i++)
    {
      CkPrintf("Iter=%d\tTime=%lf seconds\n",i,gd->itimes[i]/numparts);
    }
  }
  // CkPrintf("[%d] exiting driver\n", myid);
}

extern "C" void
finalize()
{
}

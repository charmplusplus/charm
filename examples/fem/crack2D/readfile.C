#include "crack.h"
#include <fstream.h>
#include <math.h>

#define MAXLINE 1024
#define cl(fd, buffer, n) do {\
                               for(int _i=0; _i<(n); _i++) \
                                 fd.getline(buffer, MAXLINE);\
                             } while(0)
void
readFile(GlobalData *gd)
{
  ifstream fg; // global parameter file
  ifstream fm; // mesh file
  char buf[MAXLINE];
  int i, itmp;
  double dtmp;

  fg.open("cohesive.inp");
  fm.open("crck_bar.inp");
  if (!fg || !fm)
  {
    CkAbort("Cannot open input files for reading.\n");
    return;
  }
  cl(fg,buf,3); // ignore first three lines
  fg >> gd->nTime >> gd->steps >> gd->tsintFull 
     >> gd->tsintEnergy >> gd->restart;
  cl(fg, buf, 2);
  fg >> gd->nplane >> gd->ncoh >> gd->lin;
  cl(fg, buf, 2);
  //read volumetric material properties
  fg >> gd->numMatVol;
  gd->volm = new VolMaterial[gd->numMatVol];
  VolMaterial *v;
  cl(fg, buf, 2);
  for (i=0; i<gd->numMatVol; i++)
  {
    v = &(gd->volm[i]);
    fg >> v->e1 >> v->e2 >> v->g12 >> v->g23;
    fg >> v->xnu12 >> v->xnu23 >> v->rho;
    fg >> v->alpha1 >> v->alpha2 >> v->theta;
    cl(fg, buf, 2);
  }
  //read cohesive material properties
  fg >> gd->numMatCoh;
  gd->cohm = new CohMaterial[gd->numMatCoh];
  CohMaterial *c;
  cl(fg, buf, 2);
  for (i=0; i<gd->numMatCoh; i++)
  {
    c = &(gd->cohm[i]);
    fg >> c->deltan >> c->deltat >> c->sigmax
       >> c->taumax >> c->mu;
    if (gd->ncoh)
      fg >> c->Sinit;
    cl(fg, buf, 2);
  }
  //read impact data
  fg >> gd->imp >> gd->voImp >> gd->massImp >> gd->radiusImp;
  cl(fg, buf, 2);
  fg >> gd->eImp >> gd->xnuImp;
  cl(fg, buf, 2);
  gd->voImp = 0; gd->massImp = 1.0; gd->radiusImp = 0; 
  gd->eImp = 1.0; gd->xnuImp = 0.3;
  //read thermal load
  fg >> dtmp;
  fg.close();
  fm >> itmp >> itmp >> gd->delta >> gd->numProp;
  gd->delta /= (double) gd->steps;
  gd->delta2 = gd->delta*gd->delta*0.5;
  gd->ts_proportion = new int[gd->numProp];
  gd->proportion = new double[gd->numProp];
  for (i=0; i< gd->numProp; i++) {
    fm >> gd->ts_proportion[i] >> gd->proportion[i];
  }
  //read nodal co-ordinates
  fm >> gd->numNP;
  cl(fm,buf,1);
  int curline = 0;
  for(i=0; i<gd->nn; i++)
  {
    cl(fm, buf, gd->nnums[i]-curline);
    curline = gd->nnums[i];
    Node *np = &(gd->nodes[i]);
    fm >> itmp >> np->pos.x >> np->pos.y;
    if(itmp != gd->nnums[i]+1)
    {
      CkError("[%d] Expected to read node %d got %d\n", gd->myid, 
              gd->nnums[i]+1, itmp);
      CkAbort("");
    }
    np->xM.x = np->xM.y = 0;
    np->disp.x = np->disp.y = 0.0;
    np->vel.x = np->vel.y = 0.0;
    np->accel.x = np->accel.y = 0.0;
  }
  cl(fm, buf, gd->numNP-curline);
  //read nodal boundary conditions
  fm >> gd->numBound;
  for (i=0; i<gd->numBound; i++)
  {
    int j, id0;
    fm >> id0;
    id0--;
    for (j=0; j<gd->nn && id0!=gd->nnums[j]; j++);
    if(j==gd->nn)
    {
      cl(fm, buf, 1);
      continue;
    }
    Node *np = &(gd->nodes[j]);
    np->isbnd = 1;
    fm >> np->id1 >> np->id2 >> np->r.x >> np->r.y;
  }
  //read cohesive element data, determine the length and angle
  fm >> itmp >> gd->numCLST >> itmp >> itmp >> itmp;
  cl(fm,buf,1);
  curline = 0;
  for(i=0;i<gd->ne && gd->enums[i]<gd->numCLST;i++);
  gd->scoh = 0;
  gd->ecoh = i;
  gd->svol = i;
  gd->evol = gd->ne;
  for(i=0; i<gd->ne && gd->enums[i]<gd->numCLST; i++)
  {
    cl(fm, buf, gd->enums[i]-curline);
    curline = gd->enums[i];
    Element *ep = &(gd->elements[i]);
    fm >> ep->material; ep->material--;
    int k;
    for(k=0;k<6;k++) {
      fm >> itmp;
    }
    fm >> itmp;
    Node *np1 = &(gd->nodes[gd->conn[6*i+1]]);
    Node *np2 = &(gd->nodes[gd->conn[6*i]]);
    Coh *coh = &(ep->c);
    coh->Sthresh[2] = coh->Sthresh[1] =
      coh->Sthresh[0] = gd->cohm[ep->material].Sinit;
    double x = np1->pos.x - np2->pos.x;
    double y = np1->pos.y - np2->pos.y;
    coh->sidel[0] = sqrt(x*x+y*y);
    coh->sidel[1] = x/coh->sidel[0];
    coh->sidel[2] = y/coh->sidel[0];
  }
  cl(fm, buf, gd->numCLST-curline);
  fm >> itmp >> gd->numLST >> itmp;
  cl(fm,buf,1);
  curline = gd->numCLST;
  for(gd->svol;i<gd->ne;i++)
  {
    cl(fm, buf, gd->enums[i]-curline);
    curline = gd->enums[i];
    Element *ep = &(gd->elements[i]);
    fm >> ep->material; ep->material--;
    int k;
    for(k=0;k<6;k++) fm >> itmp;
    for(k=0;k<3;k++)
    {
      ep->v.s11l[k] = 0.0;
      ep->v.s22l[k] = 0.0;
      ep->v.s12l[k] = 0.0;
    }
  }
  cl(fm, buf, gd->numCLST-curline);
  fm.close();
  vol_elem(gd);
}

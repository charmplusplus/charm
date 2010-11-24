/**
 * Read and maintain basic mesh for crack propagation code.
 */

#include "converse.h"
#include <fstream>
using namespace std;

#include <stddef.h>
#include "crack.h"

#define MAXLINE 1024
// Silly "skip n lines" macro.  Should actually only skip 
//  comment lines, but this will have to do...
#define cl(fd, buffer, n) do {\
                               for(int _i=0; _i<(n); _i++) \
                                 fd.getline(buffer, MAXLINE);\
                             } while(0)

/// Read this mesh from this file:
void
readMesh(MeshData *mesh, const char *meshFile)
{
  ifstream fm; // mesh file
  char buf[MAXLINE];
  int i, itmp;
  double dtmp;
  
  fm.open(meshFile);
  if (!fm)
  {
    crack_abort("Cannot open mesh file for reading.\n");
    return;
  }
  
  //Skip ramp-up for boundary conditions (already read by readConfig)
  int numProp;
  fm >> itmp >> itmp >> dtmp >> numProp;
  for (i=0; i< numProp; i++) {
    fm >> itmp >> dtmp;
  }
  
  //read nodal co-ordinates
  fm >> mesh->nn;
  cl(fm,buf,1);
  mesh->nodes=new Node[mesh->nn];
  for(i=0; i<mesh->nn; i++)
  {
    Node *np = &(mesh->nodes[i]);
    fm >> itmp >> np->pos.x >> np->pos.y;
    np->isbnd = 0;
  }
  
  //read nodal boundary conditions
  fm >> mesh->numBound;
  for (i=0; i<mesh->numBound; i++)
  {
    int j; // Boundary condition i applies to node j
    fm >> j;
    j--; //Fortran to C indexing
    Node *np = &(mesh->nodes[j]);
    np->isbnd = 1;
    fm >> np->id1 >> np->id2 >> np->r.x >> np->r.y;
  }
  
  //read cohesive elements
  fm >> itmp >> mesh->nc >> itmp >> itmp >> itmp;
  cl(fm,buf,1);
  mesh->cohs=new Coh[mesh->nc];
  for(i=0; i<mesh->nc; i++)
  {
    Coh *coh = &(mesh->cohs[i]);
    fm >> coh->material; coh->material--;
    int k;
    for(k=0;k<6;k++) {
      fm >> coh->conn[k]; coh->conn[k]--;
    }
    fm >> itmp; // What *is* this?
  }
  
  // Read volumetric elements
  fm >> itmp >> mesh->ne >> itmp;
  cl(fm,buf,1);
  mesh->vols=new Vol[mesh->ne];
  for(i=0;i<mesh->ne;i++)
  {
    Vol *vol = &(mesh->vols[i]);
    fm >> vol->material; vol->material--;
    int k;
    for(k=0;k<6;k++) {
      fm >> vol->conn[k]; vol->conn[k]--;
    }
  }
  fm.close();
}

void setupMesh(MeshData *mesh) {
  int i;
  //Zero out nodes
  for(i=0; i<mesh->nn; i++)
  {
    Node *np = &(mesh->nodes[i]);
    // np->pos is already set
    np->xM.x = np->xM.y = 0;
    np->disp.x = np->disp.y = 0.0;
    np->vel.x = np->vel.y = 0.0;
    np->accel.x = np->accel.y = 0.0;
  }
  
  // Init cohesive elements, determine the length and angle
  for(i=0; i<mesh->nc; i++)
  {
    Coh *coh = &(mesh->cohs[i]);
    // coh->material and coh->conn[k] are already set
    
    Node *np1 = &(mesh->nodes[coh->conn[1]]);
    Node *np2 = &(mesh->nodes[coh->conn[0]]);
    coh->Sthresh[2] = coh->Sthresh[1] =
      coh->Sthresh[0] = config.cohm[coh->material].Sinit;
    double x = np1->pos.x - np2->pos.x;
    double y = np1->pos.y - np2->pos.y;
    coh->sidel[0] = sqrt(x*x+y*y);
    coh->sidel[1] = x/coh->sidel[0];
    coh->sidel[2] = y/coh->sidel[0];
  }
  
  // Set up volumetric elements
  for(i=0;i<mesh->ne;i++)
  {
    Vol *vol = &(mesh->vols[i]);
    // vol->material and vol->conn[k] are already set
    for(int k=0;k<3;k++)
    {
      vol->s11l[k] = 0.0;
      vol->s22l[k] = 0.0;
      vol->s12l[k] = 0.0;
    }
  }
  
  // Compute the mass of local elements:
  for (i=0;i<mesh->ne;i++)
  {
    Vol *v=&(mesh->vols[i]);
    Node *n[6];              //Pointers to each of the triangle's nodes
    int k;                  //Loop index
    for (k=0;k<6;k++)
      n[k]=&(mesh->nodes[v->conn[k]]);
    //Compute the mass of this element
    double area=((n[1]->pos.x-n[0]->pos.x)*(n[2]->pos.y-n[0]->pos.y)-
                 (n[2]->pos.x-n[0]->pos.x)*(n[1]->pos.y-n[0]->pos.y));
    double mass=config.volm[v->material].rho*area/114.0;
    //Divide the element's mass among the element's nodes
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

void deleteMesh(MeshData *mesh)
{
  delete[] mesh->nodes; mesh->nodes=NULL;
  delete[] mesh->vols;  mesh->vols=NULL;
  delete[] mesh->cohs;  mesh->cohs=NULL;
}


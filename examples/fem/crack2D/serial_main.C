/**
 * Main implementation file for serial version of
 * crack propagation code.
 */
#include <stdio.h>
#include <stddef.h>
#include "crack.h"

void crack_abort(const char *why)
{
  fprintf(stderr,"Fatal error> %s\n",why);
  abort();
}

int main(int argc,char *argv[])
{
  //Read mesh data:
  printf("Reading mesh...\n");
  MeshData mesh;
  readMesh(&mesh,"crck_bar.inp");
  
  //Read our configuration data.
  readConfig("cohesive.inp","crck_bar.inp");
  
  //Set up the mesh
  setupMesh(&mesh);
  
// Prepare for the timeloop:
  NodeSlope sl;
  nodeSetup(&sl);
  int t;
  for(t=0;t<config.nTime;t++) //Timeloop:
  {
    nodeBeginStep(&mesh);
    lst_NL(&mesh);
    lst_coh2(&mesh);
    
    nodeFinishStep(&mesh, &sl, t);
    
    //For debugging, print the status of node 0:
    Node *n=&mesh.nodes[0];
    int g=0; //In a serial program, my node 0 is the true node 0.
    printf("t=%d  node=%d  d=(%g,%g)  v=(%g,%g)\n",
           t, g, n->disp.x,n->disp.y, n->vel.x,n->vel.y);
  }
}


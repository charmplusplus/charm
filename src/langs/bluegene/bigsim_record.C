
#include <stdio.h>

#include "blue.h"
#include "bigsim_record.h"
#include "blue_impl.h"

BgMessageRecorder::BgMessageRecorder(FILE * f_, int node) :f(f_) {
  nodelevel = node;
  fwrite(&nodelevel, sizeof(int),1,f);

  // write processor data
  int d = BgGetGlobalWorkerThreadID();
  int mype = d;
  fwrite(&d, sizeof(int),1,f);

  nodeInfo *myNode = cta(threadinfo)->myNode;
  d = myNode->x;
  fwrite(&d, sizeof(int), 1, f);
  d = myNode->y;
  fwrite(&d, sizeof(int), 1, f);
  d = myNode->z;
  fwrite(&d, sizeof(int), 1, f);

  d = BgGetThreadID();
  fwrite(&d, sizeof(int), 1, f);

  d = BgNumNodes();
  fwrite(&d, sizeof(int), 1, f);

  d = BgNumNodes()*BgGetNumWorkThread();
  fwrite(&d, sizeof(int), 1, f);
 
  d = cva(bgMach).x;
  fwrite(&d, sizeof(int), 1, f);
  d = cva(bgMach).y;
  fwrite(&d, sizeof(int), 1, f);
  d = cva(bgMach).z;
  fwrite(&d, sizeof(int), 1, f);
 
  if (nodelevel == 1 && mype % BgGetNumWorkThread() == 0) {
    write_nodeinfo();
  }
}

void BgMessageRecorder::write_nodeinfo()
{
  int mype = BgGetGlobalWorkerThreadID();
  int mynode = mype / BgGetNumWorkThread();
  char fName[128];
  sprintf(fName,"bgnode_%06d.log",mynode);
  FILE *fp = fopen(fName, "w");
  fprintf(fp, "%d %d\n", mype, mype + BgGetNumWorkThread()-1);
  fclose(fp);
}

BgMessageReplay::BgMessageReplay(FILE * f_, int node) :f(f_) {
  nodelevel = node;
  lcount = rcount = 0;

  int d;
  fread(&d, sizeof(int), 1, f);
  if (nodelevel != d) {
    CmiPrintf("BgReplay> Fatal error: can not replay %s logs.\n", d?"node level":"processor level");
    CmiAbort("BgReplay error");
  }

  fread(&d, sizeof(int), 1, f);
  BgSetGlobalWorkerThreadID(d);

  nodeInfo *myNode = cta(threadinfo)->myNode;
  fread(&d, sizeof(int), 1, f);
  myNode->x = d;
  fread(&d, sizeof(int), 1, f);
  myNode->y = d;
  fread(&d, sizeof(int), 1, f);
  myNode->z = d;

  fread(&d, sizeof(int), 1, f);
  BgSetThreadID(d);
  if (nodelevel == 0 && d != 0)
    CmiAbort("BgReplay> Fatal error: can not replay processor level log of rank other than 0.");
  fread(&d, sizeof(int), 1, f);
  BgSetNumNodes(d);
  fread(&d, sizeof(int), 1, f);
  if (nodelevel == 1 && d/BgNumNodes() != BgGetNumWorkThread())
    CmiAbort("BgReplay> Fatal error: the number of worker threads is not the same as in the logs.\n");
  BgSetNumWorkThread(d/BgNumNodes());

  fread(&d, sizeof(int), 1, f);
  cva(bgMach).x = d;
  fread(&d, sizeof(int), 1, f);
  cva(bgMach).y = d;
  fread(&d, sizeof(int), 1, f);
  cva(bgMach).z = d;

  //myNode->id = nodeInfo::XYZ2Local(myNode->x,myNode->y,myNode->z);

  CmiPrintf("BgMessageReplay: PE => %d NumPes => %d wth:%d\n", BgGetGlobalWorkerThreadID(), BgNumNodes()*BgGetNumWorkThread(), BgGetNumWorkThread());

  replay();
}

void BgRead_nodeinfo(int node, int &startpe, int &endpe)
{
    char fName[128];
    sprintf(fName,"bgnode_%06d.log",node);
    FILE *fp = fopen(fName, "r");
    if (fp==NULL) {
      CmiPrintf("BgReplayNode> metadata file for node %d does not exist!\n", node);
      CmiAbort("BgRead_nodeinfo");
    }
    fscanf(fp, "%d %d\n", &startpe, &endpe);
    fclose(fp);
}


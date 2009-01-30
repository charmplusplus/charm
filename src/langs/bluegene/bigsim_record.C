
#include <stdio.h>

#include "blue.h"
#include "bigsim_record.h"
#include "blue_impl.h"

BgMessageRecorder::BgMessageRecorder(FILE * f_) :f(f_) {
  // write processor data
  int d = BgGetGlobalWorkerThreadID();
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
}

BgMessageReplay::BgMessageReplay(FILE * f_) :f(f_) {
  int d;
  fread(&d, sizeof(int), 1, f);
printf("d: %d\n", d);
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
  fread(&d, sizeof(int), 1, f);
  BgSetNumNodes(d);
  fread(&d, sizeof(int), 1, f);
  BgSetNumWorkThread(d/BgNumNodes());

  fread(&d, sizeof(int), 1, f);
  cva(bgMach).x = d;
  fread(&d, sizeof(int), 1, f);
  cva(bgMach).y = d;
  fread(&d, sizeof(int), 1, f);
  cva(bgMach).z = d;

  //myNode->id = nodeInfo::XYZ2Local(myNode->x,myNode->y,myNode->z);

  CmiPrintf("BgMessageReplay: PE => %d NumPes => %d \n", BgGetGlobalWorkerThreadID(), BgNumNodes()*BgGetNumWorkThread());

  replay();
}

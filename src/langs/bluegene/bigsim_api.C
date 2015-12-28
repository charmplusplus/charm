
#include "blue.h"
#include "blue_impl.h"

// BigSim log API

int BgIsInALog(BgTimeLineRec &tlinerec)
{
  if (tlinerec.bgCurLog) return 1;
  else return 0;
}

// returns the last log in timeline
BgTimeLog *BgLastLog(BgTimeLineRec &tlinerec)
{
  CmiAssert(tlinerec.length() > 0);
  return tlinerec[tlinerec.length()-1];
}

// add deplog as backward dep of curlog, i.e. deplog <= curlog
void BgAddBackwardDep(BgTimeLog *curlog, BgTimeLog* deplog)
{
  curlog->addBackwardDep(deplog);
}

BgTimeLog *BgStartLogByName(BgTimeLineRec &tlinerec, int ep, const char *name, double starttime, BgTimeLog *prevLog)
{
  BgTimeLog* newLog = new BgTimeLog(ep, name, starttime);
  if (prevLog) {
    newLog->addBackwardDep(prevLog);
  }
  tlinerec.logEntryStart(newLog);
  return newLog;
}

void BgEndLastLog(BgTimeLineRec &tlinerec)
{
  tlinerec.logEntryClose();
}

//
// BigSim APIs for writing trace log files
//

//  dump timeline into ASCII format
void BgWriteThreadTimeLine(const char *pgm, int x, int y, int z, int th, BgTimeLine &tline)
{
  char *fname = (char *)malloc(strlen(pgm)+100);
  sprintf(fname, "%s-%d-%d-%d.%d.log", pgm, x,y,z,th);
  FILE *fp = fopen(fname, "w");
  CmiAssert(fp!=NULL);
  for (int i=0; i<tline.length(); i++) {
    fprintf(fp, "[%d] ", i);
    tline[i]->write(fp);
  }
  fclose(fp);
  free(fname);
}

// write bgTrace file:
void BgWriteTraceSummary(int numEmulatingPes, int x, int y, int z, int numWth, int numCth, const char *fname, char *traceroot)
{
  char* d = new char[512];
  BGMach bgMach;

  const PUP::machineInfo &machInfo = PUP::machineInfo::current();

  if (!traceroot) traceroot=(char*)"";
  if (fname==NULL) fname = "bgTrace";
  sprintf(d, "%s%s", traceroot,fname);
  FILE *f = fopen(d,"wb");
  if(f==NULL) {
      CmiPrintf("Creating trace file %s  failed\n", d);
      CmiAbort("BG> Abort");
  }
  PUP::toDisk p(f);
  p((char *)&machInfo, sizeof(machInfo));
  int nlocalProcs = x*y*z*numWth;               // ???
  p|nlocalProcs;
  bgMach.x = x;
  bgMach.y = y;
  bgMach.z = z;
  bgMach.numWth = numWth;
  bgMach.numCth = numCth;
  p|(BGMach &)bgMach;
  p|numEmulatingPes;
  p|bglog_version;
  int threadEP = BgLogGetThreadEP();
  if (threadEP!= -1) p|threadEP;

  printf("BgWriteTraceSummary> Number is numX:%d numY:%d numZ:%d numCth:%d numWth:%d numPes:%d totalProcs:%d bglog_ver:%d\n",bgMach.x,bgMach.y,bgMach.z,bgMach.numCth,bgMach.numWth,numEmulatingPes,nlocalProcs,bglog_version);

  fclose(f);
}

// write bgTrace<seqno> file with an array of timelines
//
// seqno:   the sequence number of the emulating processor that is responsible 
//          for generating this bgTarce?? file.
// note that target processors are mapped to bgTrace* files in round-robin 
// fashion
// that is "tlinerecs" should contains timelines of target processors of 
//   i, i+p, i+2p ..., where p is the number of emulating processors, i.e.
//   the number of bgTrace* files to write.
void BgWriteTimelines(int seqno, BgTimeLineRec **tlinerecs, int nlocalProcs, char *traceroot)
{
  int *procOffsets = new int[nlocalProcs];

  char *d = new char[512];
  sprintf(d, "%sbgTrace%d", traceroot?traceroot:"", seqno); 
  FILE *f = fopen(d,"wb");
  CmiAssert(f!=NULL);
  PUP::toDisk p(f);
  const PUP::machineInfo &machInfo = PUP::machineInfo::current();

  p((char *)&machInfo, sizeof(machInfo));	// machine info
  p|nlocalProcs;

  // CmiPrintf("Timelines are: \n");
  int procTablePos = ftell(f);
  int procTableSize = (nlocalProcs)*sizeof(int);
  fseek(f,procTableSize,SEEK_CUR); 

//  int numNodes = nlocalProcs / numWth;
  for(int i=0;i<nlocalProcs;i++) {
    BgTimeLineRec *t = tlinerecs[i];
    procOffsets[i] = ftell(f);
    t->pup(p);
  }
  
  fseek(f,procTablePos,SEEK_SET);
  p(procOffsets,nlocalProcs);
  fclose(f);

  if(CmiMyPe() == 0) 
    CmiPrintf("BgWriteTimelines> Wrote to disk for %d simulated nodes on PE0. \n", nlocalProcs);
  delete [] procOffsets;
  delete [] d;
}

void BgWriteTimelines(int seqno, BgTimeLineRec *tlinerecs, int nlocalProcs, char *traceroot)
{
  BgTimeLineRec **tlines = new BgTimeLineRec*[nlocalProcs];
  for (int i=0; i<nlocalProcs; i++)
    tlines[i] = &tlinerecs[i];

  BgWriteTimelines(seqno, tlines, nlocalProcs, traceroot);
  delete [] tlines;
}



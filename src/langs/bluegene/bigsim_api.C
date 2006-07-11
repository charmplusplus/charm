
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

BgTimeLog *BgStartLogByName(BgTimeLineRec &tlinerec, int ep, char *name, double starttime, BgTimeLog *prevLog)
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

void BgWriteThreadTimeLine(char *pgm, int x, int y, int z, int th, BgTimeLine &tline)
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


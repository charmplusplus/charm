// File: stats.C
// Modest statistics gathering facility.
// Last Modified: 7.31.01 by Terry L. Wilmarth

#include "charm++.h"
#include "stats.h"
#include "stats.def.h"

CkChareID theGlobalStats;
CkGroupID theLocalStats;

extern void POSE_exit();

// Send local stats to global collector
void localStat::SendStats()
{
  CProxy_globalStat gstat(theGlobalStats);
  localStatSummary *m;
  m = new localStatSummary;
  m->doTime = totalTime;
  m->rbTime = rollbackTime;
  m->gvtTime = gvtTime;
  m->simTime = simTime;
  m->cpTime = cpTime;
  m->canTime = canTime;
  m->lbTime = lbTime;
  m->miscTime = miscTime;
  m->cpBytes = cpBytes;
  m->maxDo = maxDo;
  m->minDo = minDo;
  m->pe = CkMyPe();
  m->dos = dos;
  m->undos = undos;
  m->gvts = gvts;
  m->maxChkPts = maxChkPts;
  gstat.localStatReport(m);
}

// Basic constructor initializes all accumulators to 0. 
globalStat::globalStat(void)
{
  doAvg = doMax = rbAvg = rbMax = gvtAvg = gvtMax = simAvg = simMax = 
    cpAvg = cpMax = canAvg = canMax = lbAvg = lbMax = miscAvg = miscMax = 
    maxTime = maxDo = minDo = avgDo = GvtTime = 0.0; 
  cpBytes = reporting = totalDos = totalUndos = totalGvts = maxChkPts = 0;
}

// Receive, calculate and print statistics
void globalStat::localStatReport(localStatSummary *m)
{
  double tmpMax;
  
  // accumulate data from local stats collectors
  totalDos += m->dos;
  totalUndos += m->undos;
  totalGvts += m->gvts;
  maxChkPts += m->maxChkPts;
  doAvg += m->doTime;
  if (m->maxDo > maxDo)
    maxDo = m->maxDo;
  if ((minDo < 0.0) || (m->minDo < minDo))
    minDo = m->minDo;
  if (m->doTime > doMax)
    doMax = m->doTime;
  rbAvg += m->rbTime;
  if (m->rbTime > rbMax)
    rbMax = m->rbTime;
  gvtAvg += m->gvtTime;
  if (m->gvtTime > gvtMax)
    gvtMax = m->gvtTime;
  simAvg += m->simTime;
  if (m->simTime > simMax)
    simMax = m->simTime;
  cpAvg += m->cpTime;
  if (m->cpTime > cpMax)
    cpMax = m->cpTime;
  canAvg += m->canTime;
  if (m->canTime > canMax)
    canMax = m->canTime;
  lbAvg += m->lbTime;
  if (m->lbTime > lbMax)
    lbMax = m->lbTime;
  miscAvg += m->miscTime;
  if (m->miscTime > miscMax)
    miscMax = m->miscTime;
  cpBytes += m->cpBytes;
  tmpMax = m->doTime + m->rbTime + m->gvtTime + m->simTime + m->cpTime 
    + m->canTime + m->lbTime + m->miscTime;
  CkFreeMsg(m);
  if (tmpMax > maxTime)
    maxTime = tmpMax;
  reporting++;

  GvtTime = gvtAvg/totalGvts;
  if (reporting == CkNumPes()) { // all local stats are in
    // compute final values
    avgDo = doAvg / totalDos;
    doAvg /= CkNumPes();
    rbAvg /= CkNumPes();
    gvtAvg /= CkNumPes();
    simAvg /= CkNumPes();
    cpAvg /= CkNumPes();
    canAvg /= CkNumPes();
    lbAvg /= CkNumPes();
    miscAvg /= CkNumPes();
    maxChkPts /= CkNumPes();
    // print stats table (all one print to avoid breaking up)
    CkPrintf("----------------------------------------------------------------------------\n   | DO     | RB     | GVT    | SIM    | CP     | CAN    | LB     | MISC   |\n---|--------|--------|--------|--------|--------|--------|--------|--------|\nmax| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f|\navg| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f|\n----------------------------------------------------------------------------\nMax time on a PE: %7.2f,  Speculative Events: %d Actual Events: %d\nGRAINSIZE INFO: Avg: %10.6f Max: %10.6f Min: %10.6f\nGVT iterations=%d  Avg time per iteration=%f  Avg. Max# Checkpoints=%d\n", 
	     doMax, rbMax, gvtMax, simMax, cpMax, canMax, lbMax, miscMax, doAvg, rbAvg, gvtAvg, simAvg, cpAvg, canAvg, lbAvg, miscAvg, maxTime, totalDos, totalDos-totalUndos, avgDo, maxDo, minDo, totalGvts, GvtTime, maxChkPts);
    POSE_exit();
  }
}



/// Modest statistics gathering facility for POSE
#include "charm++.h"
#include "stats.h"
#include "stats.def.h"

CkChareID theGlobalStats;
CkGroupID theLocalStats;

extern void POSE_exit();

/// Send local stats to global collector
void localStat::SendStats()
{
  if (pose_config.dop) {
    // This ensures everything is flushed to the file
    fclose(dopFilePtr);
    // Right now, SendStats is called only at the end of the
    // simulation.  This is here in case that changes for some reason.
    dopFilePtr = fopen(dopFileName, "a");
    if (dopFilePtr == NULL) {
      CkPrintf("WARNING: unable to open DOP file %s for append...this probably doesn't matter, though, as long as this is at the end of the simulation\n", 
	       dopFileName);
    }
  }
  CProxy_globalStat gstat(theGlobalStats);
  localStatSummary *m = new localStatSummary;
  m->doTime = totalTime;
  m->rbTime = rollbackTime;
  m->gvtTime = gvtTime;
  m->simTime = simTime;
  m->cpTime = cpTime;
  m->canTime = canTime;
  m->lbTime = lbTime;
  m->fcTime = fcTime;
  m->commTime = commTime;
  m->cpBytes = cpBytes;
  m->maxDo = maxDo;
  m->minDo = minDo;
  m->pe = CkMyPe();
  m->dos = dos;
  m->undos = undos;
  m->commits = commits;
  m->loops = loops;
  m->gvts = gvts;
  m->maxChkPts = maxChkPts;
  m->maxGVT = maxGVT;
  m->maxGRT = maxGRT;
  gstat.localStatReport(m);
}

// Basic Constructor
globalStat::globalStat(void):    doAvg (0.0), doMax (0.0), rbAvg (0.0), rbMax (0.0), gvtAvg (0.0), gvtMax (0.0), simAvg (0.0), simMax (0.0), 
    cpAvg (0.0), cpMax (0.0), canAvg (0.0), canMax (0.0), lbAvg (0.0), lbMax (0.0), fcAvg (0.0), fcMax (0.0), 
    commAvg (0.0), commMax (0.0),
    maxTime (0.0), maxDo (0.0), minDo (0.0), avgDo (0.0), GvtTime (0.0), maxGRT (0.0),
  cpBytes (0), reporting (0), totalDos (0), totalUndos (0), totalCommits (0), totalLoops (0), 
    totalGvts (0), maxChkPts (0), maxGVT (0)
{
#ifdef VERBOSE_DEBUG
  CkPrintf("[%d] constructing globalStat\n",CkMyPe());
#endif

}

// Receive, calculate and print statistics
void globalStat::localStatReport(localStatSummary *m)
{
  double tmpMax;
  // accumulate data from local stats collectors
  totalDos += m->dos;
  totalUndos += m->undos;
  totalCommits += m->commits;
  totalLoops += m->loops;
  totalGvts += m->gvts;
  maxChkPts += m->maxChkPts;
  doAvg += m->doTime;
  if (maxGRT < m->maxGRT) maxGRT = m->maxGRT;
  if (maxGVT < m->maxGVT) maxGVT = m->maxGVT;
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
  fcAvg += m->fcTime;
  if (m->fcTime > fcMax)
    fcMax = m->fcTime;
  commAvg += m->commTime;
  if (m->commTime > commMax)
    commMax = m->commTime;
  cpBytes += m->cpBytes;
  tmpMax = m->doTime + m->rbTime + m->gvtTime + m->simTime + m->cpTime 
    + m->canTime + m->lbTime + m->fcTime + m->commTime;
  CkFreeMsg(m);
  if (tmpMax > maxTime)
    maxTime = tmpMax;
  reporting++;

#ifdef SEQUENTIAL_POSE
  totalLoops = totalDos;
  totalGvts = 1;
#endif
  CkAssert(totalGvts > 0);
  CkAssert(totalDos > 0);
  CkAssert(totalLoops > 0);
  GvtTime = gvtAvg/totalGvts;
  if (reporting == CkNumPes()) { //all local stats are in; compute final values
    avgDo = doAvg / totalDos;
    doAvg /= CkNumPes();
    rbAvg /= CkNumPes();
    gvtAvg /= CkNumPes();
    simAvg /= CkNumPes();
    cpAvg /= CkNumPes();
    canAvg /= CkNumPes();
    lbAvg /= CkNumPes();
    fcAvg /= CkNumPes();
    commAvg /= CkNumPes();
    maxChkPts /= CkNumPes();
    // print stats table (all one print to avoid breaking up)
    CkPrintf("------------------------------------------------------------------------------------\n   | DO     | RB     | GVT    | SIM    | CP     | CAN    | LB     | FC     | COMM   |\n---|--------|--------|--------|--------|--------|--------|--------|--------|--------|\nmax| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f|\navg| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f| %7.2f|\n------------------------------------------------------------------------------------\nMax time on a PE:%7.2f, Speculative Events:%d Actual Events:%d Commits:%d\nGRAINSIZE INFO: Avg: %10.6f Max: %10.6f Min: %10.6f\nGVT iterations=%d  Avg time per iteration=%f\ntotalLoops=%d effectiveGS=%10.6f\n", 
	     doMax, rbMax, gvtMax, simMax, cpMax, canMax, lbMax, fcMax, commMax, doAvg, rbAvg, gvtAvg, simAvg, cpAvg, canAvg, lbAvg, fcAvg, commAvg, maxTime, totalDos, totalDos-totalUndos, totalCommits, avgDo, maxDo, minDo, totalGvts, GvtTime, totalLoops, (doAvg*CkNumPes())/totalLoops);
    //CkPrintf("Avg. Max# Checkpoints=%d Bytes checkpointed=%d\n", maxChkPts, cpBytes);

    if(pose_config.dop){
      CkPrintf("Overhead-free maximally parallel runtime=%f  Max GVT=%lld\n", maxGRT, maxGVT);
      if (pose_config.dopSkipCalcs) {
	CkPrintf("\n");
	CkPrintf("WARNING: Skipping DOP calculations.  Not writing dop_mod.out or dop_sim.out files.\n");
	CkPrintf("Save the following parameters for use with the DOPCalc tool in the\n");
	CkPrintf("BigNetSim/trunk/tools/DOPCalc directory of the BigNetSim svn repository:\n");
	CkPrintf("   #PEs=%d, maxGRT=%f, maxGVT=%lld, and the starting virtual time\n", CkNumPes(), maxGRT, maxGVT);
	CkPrintf("   of the simulation (which should be 0 unless there are skip points)\n");
	CkPrintf("\n");
      } else {
	DOPcalc(maxGVT, maxGRT);
      }
    }

    POSE_exit();
    /*
    reporting = 0;
    doAvg = doMax = rbAvg = rbMax = gvtAvg = gvtMax = simAvg = simMax = 
      cpAvg = cpMax = canAvg = canMax = lbAvg = lbMax = fcAvg = fcMax = 
      commAvg, commMax, maxTime = maxDo = minDo = avgDo = GvtTime = 0.0; 
    cpBytes = reporting = totalDos = totalUndos = totalCommits = totalLoops = 
      totalGvts = maxChkPts = 0;
    */
  }
}


void globalStat::DOPcalc(POSE_TimeType gvt, double grt)
{
#if CMK_LONG_LONG_DEFINED
  long long int i, j;
#else
  CmiInt8 i,j;
#endif
 
  POSE_TimeType vinStart, vinEnd, gvtp = gvt + 1;
  double rinStart, rinEnd;
  FILE *fp;
  char filename[20], line[80];
  unsigned short int *gvtDOP, *grtDOP;
  unsigned long int grtp = (unsigned long int)(grt*1000000.0) + 1, usStart, usEnd;
  int modelPEs=0, simulationPEs=0;

  CkPrintf("Generating DOP measures...\n");
  gvtDOP = (unsigned short int *)malloc(gvtp*sizeof(unsigned short int));
  grtDOP = (unsigned short int *)malloc(grtp*sizeof(unsigned short int));
  for (i=0; i<gvtp; i++) gvtDOP[i] = 0;
  for (i=0; i<grtp; i++) grtDOP[i] = 0;
  for (i=0; i<CkNumPes(); i++) { // read each processor's log
    sprintf(filename, "dop%lld.log\0", i);
    fp = fopen(filename, "r");
    if (!fp) {
      CkPrintf("Cannot open file %s... exiting.\n", filename);
      free(gvtDOP);
      free(grtDOP);
      POSE_exit();
      return;
    }
    CkPrintf("Reading file %s...\n", filename);
#if USE_LONG_TIMESTAMPS
    const char* format = "%lf %lf %lld %lld\n";
#else
    const char* format = "%lf %lf %d %d\n";
#endif
    while (fgets(line, 80, fp)) {
      if (sscanf(line, format, &rinStart, &rinEnd, &vinStart, &vinEnd) == 4) {
	usStart = (unsigned long int)(rinStart * 1000000.0);
	usEnd = (unsigned long int)(rinEnd * 1000000.0);
	for (j=usStart; j<usEnd; j++) grtDOP[j]++;
	if (usStart == usEnd) grtDOP[usStart]++;
	if (vinStart > -1)
	  for (j=vinStart; j<=vinEnd; j++) gvtDOP[j]++;
      }
      else CkPrintf("WARNING: DOP post-processing corrupted... likely NFS to blame.\n");
    }
    fclose(fp);
  }
  CmiInt8 avgPEs = 0LL;
  int zed = 0;
  fp = fopen("dop_mod.out", "w");
  for (i=0; i<gvtp; i++) {
    // print status every ~64M iterations
    if ((i & 0x03FFFFFF) == 0) CkPrintf("   current index: %lld of %lld\n", i, gvtp);
    if ((gvtDOP[i] != 0) || (zed == 0))
      fprintf(fp, "%lld %d\n", i, gvtDOP[i]);
    if (gvtDOP[i] == 0) zed = 1;
    else zed = 0;
    avgPEs += gvtDOP[i];
    if (gvtDOP[i] > modelPEs) modelPEs = gvtDOP[i];
  }
  avgPEs /= gvtp;
  fclose(fp);
  fp = fopen("dop_sim.out", "w");
  for (i=0; i<grtp; i++) {
    fprintf(fp, "%lld %d\n", i, grtDOP[i]);
    if (grtDOP[i] > simulationPEs) simulationPEs = grtDOP[i];
  } 
  fclose(fp);
  CkPrintf("Max model PEs: %d  Max simulation PEs: %d  Recommended #PEs: %lld\n",
	   modelPEs, simulationPEs, avgPEs);
  free(gvtDOP);
  free(grtDOP);
 
}

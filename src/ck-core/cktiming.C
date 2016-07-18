#include "charm++.h"
#include "cktiming.h"
#include <stdio.h>
#include <stdarg.h>
#include <vector>

/*
 *
 * WITH_MAMBO:    when using mambo
 * CMK_BIGSIM_CHARM:  when build with bigsim
 * SPLIT_APART_CYCLE_ACCURATE: parallelize Mambo run
 * */

#undef WITH_MAMBO 
#undef SPLIT_APART_CYCLE_ACCURATE

// MAMBO
#if  WITH_MAMBO
#include "mambo.h"
static uint64_t start_time, end_time;
#else
CkpvStaticDeclare(double, start_time);
CkpvStaticDeclare(double, end_time);
#endif

CkpvDeclare(FILE*, bgfp);     // for bigsim run

CkpvDeclare(int, outputParameters);




//======================PAPI======================= 
//#define BIG_SIM_PAPI
#ifdef BIG_SIM_PAPI
#include <papi.h>
#define NUM_PAPI_EVENTS 9
#define BIGSIM_PAPI

int errorcode; 
int events[NUM_PAPI_EVENTS];
long long values[NUM_PAPI_EVENTS]; 
char errorstring[PAPI_MAX_STR_LEN+1]; 
#endif

unsigned long bgTraceCounter;
double startTime;

CkpvStaticDeclare(bool, insideTraceBracket);

class StringPool {
 std::vector<char *> events;
 int dumped;
public:
  StringPool(): dumped(0) {}
  void dump() {
    char fname[128];
    const char *subdir = "params";
    if (dumped) return;
    CmiMkdir(subdir);
    sprintf(fname, "%s/param.%d", subdir, CkMyPe());
    FILE *fp  = fopen(fname, "w");
    if (fp == NULL) 
      CmiAbort("Failed to generated trace param file!");
      // write out
    for (int i=0; i<events.size(); i++)
      fprintf(fp, "%s", events[i]);
    fclose(fp);
    dumped = 1;
  }
  void insert(char *e) {
    events.push_back(strdup(e));
  }
};

CkpvStaticDeclare(StringPool, eventsPool);

#if CMK_BIGSIM_CHARM
static int outputTiming = 0;
#endif

// called on all PEs once
extern "C"
void initBigSimTrace(int outputParams, int _outputTiming)
{
  CkpvInitialize(int, outputParameters);
  CkpvAccess(outputParameters) = outputParams;

  bgTraceCounter = 0;
#if CMK_BIGSIM_CHARM
  if (!BgIsReplay()) outputTiming = 0;
  outputTiming = _outputTiming;
#endif
  CkpvInitialize(bool, insideTraceBracket);
  CkpvAccess(insideTraceBracket) = false;

  CkpvInitialize(double, start_time);
  CkpvInitialize(double, end_time);

  CkpvInitialize(FILE*, bgfp);
  CkpvAccess(bgfp) = NULL;
#if CMK_BIGSIM_CHARM
  //   for bigsim emulation, write to files, one for each processor
  //   always write immediately, instead of store and dump at the end
  if (!outputTiming) {
  char fname[128];
  const char *subdir = "params";
  CmiMkdir(subdir);
  sprintf(fname, "%s/param.%d", subdir, CkMyPe());
  CkpvAccess(bgfp) = fopen(fname, "w");
  if (CkpvAccess(bgfp) == NULL) 
    CmiAbort("Failed to generated trace param file!");
  }
#endif
  //   for Mambo simulation, write to screen for now
//  CkpvAccess(bgfp) = stdout;
  if (CkpvAccess(outputParameters))  { 
    CkpvInitialize(StringPool, eventsPool);
    if (CkMyPe()==0) CmiPrintf("outputParameters enabled!\n");
#if CMK_BIGSIM_CHARM
    BgRegisterUserTracingFunction(finalizeBigSimTrace);
#endif
  }


#ifdef BIG_SIM_PAPI
	CkPrintf("PAPI: number of available counters: %d\n", PAPI_num_counters());
	CkAssert(PAPI_num_counters() >= 0);
#endif

}

extern "C"
void finalizeBigSimTrace()
{
  if (CkpvAccess(bgfp) != NULL) {
    fclose(CkpvAccess(bgfp));
    CkpvAccess(bgfp) = NULL;
    CkpvAccess(outputParameters) = 0;
  }
  else {
    if (CkpvAccess(outputParameters))
      CkpvAccess(eventsPool).dump();
  }
}

extern "C"
void startTraceBigSim(){

  CkAssert(CkpvAccess(insideTraceBracket) == false);
  CkpvAccess(insideTraceBracket) = true;

#if SPLIT_APART_CYCLE_ACCURATE
  SimParameters *simParams = Node::Object()->simParameters;
  if(simParams->bgSplitNumProcs != -1 && simParams->bgSplitMyProc!=-1){
	(bgTraceCounter) ++;
	if( ((bgTraceCounter) % simParams->bgSplitNumProcs) == simParams->bgSplitMyProc){
	  // Do slow mambo simulation for this case!
	  //CkPrintf("TRACEBIGSIM: Doing cycle accurate simulation for interesting event #%lu\n", (bgTraceCounter) );
	  start_time = begin(); // for MAMBO
	}
  }
#endif  


#ifdef BIGSIM_PAPI

	for(int i=0;i<NUM_PAPI_EVENTS;i++)
		values[i] = 0;

	events[0] = PAPI_FP_OPS;
	events[1] = PAPI_TOT_INS;
	events[2] = PAPI_L1_ICM;
	events[3] = PAPI_L2_TCM;
	events[4] = PAPI_L3_TCM;
	events[5] = PAPI_TLB_TL;
	events[6] = PAPI_LD_INS;
	events[7] = PAPI_SR_INS; // store instructions
	events[8] = PAPI_RES_STL; // resource stalls

/* Other available events:
					PAPI_BR_INS,
                                        PAPI_BR_MSP,
                                        PAPI_FP_INS,
                                        PAPI_TOT_INS,
                                        PAPI_TOT_IIS,
                                        PAPI_L1_DCM,
                                        PAPI_L1_LDM,
                                        PAPI_L2_TCM,
                                        PAPI_L3_LDM,
                                        PAPI_RES_STL,
                                        PAPI_LD_INS,
                                        PAPI_TLB_TL  
*/

	CkAssert(PAPI_start_counters(events, NUM_PAPI_EVENTS) == PAPI_OK);

#endif


#if CMK_BIGSIM_CHARM
  BgMark("startTraceBigSim %f\n");
#endif

#if WITH_MAMBO
  //	startTime = CmiWallTimer();
  start_time = begin(); // for MAMBO
#else
  CkpvAccess(start_time) = CmiWallTimer();
#endif
  
}
  

extern "C"
void endTraceBigSim_20param(char * eventname, int stepno, int num_params, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 , double p16 , double p17 , double p18 , double p19 , double p20 ) {

#if WITH_MAMBO
    end_time=end();
	 //	double endTime = CmiWallTimer();
#else
    CkpvAccess(end_time) = CmiWallTimer();
#endif

    CkAssert(CkpvAccess(insideTraceBracket) == true);
    CkpvAccess(insideTraceBracket) = false;
#if CMK_BIGSIM_CHARM
    char perfCountString[1024]; 
    perfCountString[0] = 0; 
#endif
	char params[2048];

if(num_params==0) sprintf(params, "");
if(num_params==1) sprintf(params, "%f", p1);
if(num_params==2) sprintf(params, "%f %f", p1, p2);
if(num_params==3) sprintf(params, "%f %f %f", p1, p2, p3);
if(num_params==4) sprintf(params, "%f %f %f %f", p1, p2, p3, p4);
if(num_params==5) sprintf(params, "%f %f %f %f %f", p1, p2, p3, p4, p5);
if(num_params==6) sprintf(params, "%f %f %f %f %f %f", p1, p2, p3, p4, p5, p6);
if(num_params==7) sprintf(params, "%f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7);
if(num_params==8) sprintf(params, "%f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8);
if(num_params==9) sprintf(params, "%f %f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8, p9);
if(num_params==10) sprintf(params, "%f %f %f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
if(num_params==11) sprintf(params, "%f %f %f %f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
if(num_params==12) sprintf(params, "%f %f %f %f %f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);
if(num_params==13) sprintf(params, "%f %f %f %f %f %f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
if(num_params==14) sprintf(params, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14);
if(num_params==15) sprintf(params, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15);
if(num_params==16) sprintf(params, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);
if(num_params==17) sprintf(params, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17);
if(num_params==18) sprintf(params, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18);
if(num_params==19) sprintf(params, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19);
if(num_params==20) sprintf(params, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20);

	char paramString[2048];
	sprintf(paramString, "params:{ %s }", params);
	
	char eventNameString[1024];
	sprintf(eventNameString, "event:{ %s }", eventname);
 
#ifdef BIGSIM_PAPI
        CkAssert(PAPI_stop_counters(values, NUM_PAPI_EVENTS) == PAPI_OK); 

        sprintf(perfCountString, " PAPI:{ " ); 

	for(int i=0;i<NUM_PAPI_EVENTS;i++){
		sprintf(perfCountString+strlen(perfCountString), " %lld ", values[i] );
	}


	 printf("value=%lld\n", values[0]);

	 sprintf(perfCountString+strlen(perfCountString), " }");
	
#endif

 
  char timeString[512];
  timeString[0] = 0;
  char stepString[128];
  stepString[0] = 0;
  sprintf(stepString, "step:{ %d }", stepno);

#if ! CMK_BIGSIM_CHARM
#if WITH_MAMBO
  //  sprintf(timeString, "time:{ %f }", endTime-startTime);
  sprintf(timeString, "time_in_cycles:{ %llu }",  end_time-start_time); 
#endif
#endif

  if (CkpvAccess(bgfp) == NULL) {
  if (CkpvAccess(outputParameters)) {
  double t = CkpvAccess(end_time)-CkpvAccess(start_time);
if (t<0.0) {
    CmiPrintf("time: %f\n", t);
    t = 0.0;
}
  CmiAssert(t >= 0.0);

  sprintf(timeString, "time_in_us:{ %lf } %s %s %s\n",  t*1e6, eventNameString, stepString, paramString);
  CkpvAccess(eventsPool).insert(timeString);
  }
  }


#if SPLIT_APART_CYCLE_ACCURATE
  SimParameters *simParams = Node::Object()->simParameters;
	  if(simParams->bgSplitNumProcs != -1 && simParams->bgSplitMyProc!=-1){
	if( ((bgTraceCounter) % simParams->bgSplitNumProcs) == simParams->bgSplitMyProc){
	  // Do slow mambo simulation for this case!
	  // Counter is incremented only in startTraceBigSim()
	}
  }
#endif
#if CMK_BIGSIM_CHARM

  char sequenceString[128];
  sequenceString[0] = 0;

  BgMark("endTraceBigSim %f\n");
  if (CkpvAccess(bgfp) != NULL) {
  // write event ID
  int seqno = tTIMELINEREC.length()-1;
  if (seqno<0) CkAbort("Traces are not generated. Please run emulation with +bglog");
  fprintf(CkpvAccess(bgfp),"%d ",seqno);
  sprintf(sequenceString, "seqno:{ %d } ",seqno);
//  fprintf(CkpvAccess(bgfp),"%s\n",params);
  fprintf(CkpvAccess(bgfp), "TRACEBIGSIM: %s %s %s %s %s %s\n", eventNameString, stepString, sequenceString, timeString, perfCountString, paramString);
  }
#else
/*
//  printf("TRACEBIGSIM: %s %s %s %s %s\n", eventNameString, sequenceString, timeString, perfCountString, paramString);
  if (CkpvAccess(bgfp) != NULL) {
  fprintf(CkpvAccess(bgfp), "TRACEBIGSIM: %s %s %s %s %s\n", eventNameString, sequenceString, timeString, perfCountString, paramString);
  }
*/
#endif


}





/** 
	startTraceBigSim() begins tracing an event with a specified set of parameters. These are convenience
	aliases so that a user can easily add/remove parameters while testing their application.
	Up to 20 parameters can be specified. These just call through to startTraceBigSim_20param().
*/
void endTraceBigSim( char * eventName, int stepno ){endTraceBigSim_20param( eventName, stepno, 0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 ){endTraceBigSim_20param( eventName, stepno, 1 , p1 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 ){endTraceBigSim_20param( eventName, stepno, 2 , p1 , p2 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 ){endTraceBigSim_20param( eventName, stepno, 3 , p1 , p2 , p3 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 ){endTraceBigSim_20param( eventName, stepno, 4 , p1 , p2 , p3 , p4 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 ){endTraceBigSim_20param( eventName, stepno, 5 , p1 , p2 , p3 , p4 , p5 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 ){endTraceBigSim_20param( eventName, stepno, 6 , p1 , p2 , p3 , p4 , p5 , p6 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 ){endTraceBigSim_20param( eventName, stepno, 7 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 ){endTraceBigSim_20param( eventName, stepno, 8 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 ){endTraceBigSim_20param( eventName, stepno, 9 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , p9 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 ){endTraceBigSim_20param( eventName, stepno, 10 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , p9 , p10 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 ){endTraceBigSim_20param( eventName, stepno, 11 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , p9 , p10 , p11 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 ){endTraceBigSim_20param( eventName, stepno, 12 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , p9 , p10 , p11 , p12 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 ){endTraceBigSim_20param( eventName, stepno, 13 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , p9 , p10 , p11 , p12 , p13 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 ){endTraceBigSim_20param( eventName, stepno, 14 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , p9 , p10 , p11 , p12 , p13 , p14 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 ){endTraceBigSim_20param( eventName, stepno, 15 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , p9 , p10 , p11 , p12 , p13 , p14 , p15 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 , double p16 ){endTraceBigSim_20param( eventName, stepno, 16 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , p9 , p10 , p11 , p12 , p13 , p14 , p15 , p16 , 0.0 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 , double p16 , double p17 ){endTraceBigSim_20param( eventName, stepno, 17 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , p9 , p10 , p11 , p12 , p13 , p14 , p15 , p16 , p17 , 0.0 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 , double p16 , double p17 , double p18 ){endTraceBigSim_20param( eventName, stepno, 18 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , p9 , p10 , p11 , p12 , p13 , p14 , p15 , p16 , p17 , p18 , 0.0 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 , double p16 , double p17 , double p18 , double p19 ){endTraceBigSim_20param( eventName, stepno, 19 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , p9 , p10 , p11 , p12 , p13 , p14 , p15 , p16 , p17 , p18 , p19 , 0.0 );}
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 , double p16 , double p17 , double p18 , double p19 , double p20 ){endTraceBigSim_20param( eventName, stepno, 20 , p1 , p2 , p3 , p4 , p5 , p6 , p7 , p8 , p9 , p10 , p11 , p12 , p13 , p14 , p15 , p16 , p17 , p18 , p19 , p20 );}

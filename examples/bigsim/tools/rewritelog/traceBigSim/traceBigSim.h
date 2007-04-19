/**
 *  @author Isaac Dooley
 *  
 *  This version is quite different than the initial version.
 */	

// CMK_BLUEGENE_CHARM is used during the emulation phase, 
// where you genereate the bgTrace logs and the parameter files

#ifdef CMK_BLUEGENE_CHARM
#include "blue.h"
#include "blue_impl.h"
#endif
#include <stdio.h>


CkpvExtern(FILE *, bgfp);

extern void initBigSimTrace();
extern void finalizeBigSimTrace();

CkpvExtern(unsigned long, bgTraceCounter);

#endif

void initBigSimTrace();
void finalizeBigSimTrace();
void BgSetStartEvent();


//======================PAPI======================= 
#define BIG_SIM_PAPI

#ifdef BIG_SIM_PAPI

#include <papi.h>

#define NUM_PAPI_EVENTS 3
#define BIGSIM_PAPI

int errorcode; 
int events[NUM_PAPI_EVENTS];
long long values[NUM_PAPI_EVENTS]; 
char errorstring[PAPI_MAX_STR_LEN+1]; 

#endif
//================================================= 


/** @TODO, wrap this with Ckpv */
double startTime;
char paramString[1024];


/** A function that starts the bigsim tracing processes with up to 20 parameters. The user should use one of the 20 aliases below which takes the right number of parameters. */
void startTraceBigSim_20param( int num_params, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 , double p16 , double p17 , double p18 , double p19 , double p20 ) 
{

  sprintf(paramString, "params:{ count=%d}", num_params);

  SimParameters *simParams = Node::Object()->simParameters;
  if(simParams->bgSplitNumProcs != -1 && simParams->bgSplitMyProc!=-1){
	if( (CkpvAccess(bgTraceCounter) % simParams->bgSplitNumProcs) == simParams->bgSplitMyProc){
	  // Do slow mambo simulation for this case!
	  CkPrintf("TRACEBIGSIM: Doing cycle accurate simulation for interesting event #%lu\n", CkpvAccess(bgTraceCounter) );
	}
	CkpvAccess(bgTraceCounter) ++;
  }
  
  startTime = CmiWallTimer();


#ifdef BIGSIM_PAPI
    events[0] = PAPI_TOT_CYC;
    events[1] = PAPI_TLB_TL;
    events[2] = PAPI_FP_INS;

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

    errorcode = PAPI_start_counters(events, NUM_PAPI_EVENTS);
    if (errorcode != PAPI_OK) {
        PAPI_perror(errorcode, errorstring, PAPI_MAX_STR_LEN);
        fprintf(stderr, "PAPI error after start_counters (%d): %s\n", errorcode, errorstring);
    }

#endif
  
}
  


/** 
	startTraceBigSim() begins tracing an event with a specified set of parameters. These are convenience
	aliases so that a user can easily add/remove parameters while testing their application.
	Up to 20 parameters can be specified. These just call through to startTraceBigSim_20param().
*/
void startTraceBigSim()   
	{startTraceBigSim_20param(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1)   
	{startTraceBigSim_20param(1, p1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2)   
	{startTraceBigSim_20param(2, p1, p2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3)   
	{startTraceBigSim_20param(3, p1, p2, p3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4)   
	{startTraceBigSim_20param(4, p1, p2, p3, p4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5)   
	{startTraceBigSim_20param(5, p1, p2, p3, p4, p5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6)   
	{startTraceBigSim_20param(6, p1, p2, p3, p4, p5, p6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7)   
	{startTraceBigSim_20param(7, p1, p2, p3, p4, p5, p6, p7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8)   
	{startTraceBigSim_20param(8, p1, p2, p3, p4, p5, p6, p7, p8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9)   
	{startTraceBigSim_20param(9, p1, p2, p3, p4, p5, p6, p7, p8, p9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10)   
	{startTraceBigSim_20param(10, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11)   
	{startTraceBigSim_20param(11, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12)   
	{startTraceBigSim_20param(12, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13)   
	{startTraceBigSim_20param(13, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14)   
	{startTraceBigSim_20param(14, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15)   
	{startTraceBigSim_20param(15, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, 0.0, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16)   
	{startTraceBigSim_20param(16, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, 0.0, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16, double p17)   
	{startTraceBigSim_20param(17, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, 0.0, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16, double p17, double p18)   
	{startTraceBigSim_20param(18, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, 0.0, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16, double p17, double p18, double p19)   
	{startTraceBigSim_20param(19, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, 0.0);}
void startTraceBigSim(double p1, double p2, double p3, double p4, double p5, double p6, double p7, double p8, double p9, double p10, double p11, double p12, double p13, double p14, double p15, double p16, double p17, double p18, double p19, double p20)   
	{startTraceBigSim_20param(20, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20);}






void endTraceBigSim(char * eventname) {
    char perfCountString[1024]; 
    perfCountString[0] = 0; 
 
#ifdef BIGSIM_PAPI
    errorcode = PAPI_read_counters(values, NUM_PAPI_EVENTS); 
    if (errorcode != PAPI_OK) { 
        PAPI_perror(errorcode, errorstring, PAPI_MAX_STR_LEN); 
        fprintf(stderr, "PAPI error after read counters (%d): %s\n", errorcode, errorstring); 
    } else { 
        sprintf(perfCountString, 
                " PAPI:{ %lld %lld %lld }", 
                values[0], 
                values[1],
		values[2] 
                ); 
    } 

    errorcode = PAPI_stop_counters(values, NUM_PAPI_EVENTS);
    if (errorcode != PAPI_OK) { 
        PAPI_perror(errorcode, errorstring, PAPI_MAX_STR_LEN); 
        fprintf(stderr, "PAPI error after stop counters (%d): %s\n", errorcode, errorstring); 
    } 
#endif
	double endTime = CmiWallTimer();
 
  char timeString[256];
  sprintf(timeString, "time:{ %f }", endTime-startTime);

  SimParameters *simParams = Node::Object()->simParameters;
  if(simParams->bgSplitNumProcs != -1 && simParams->bgSplitMyProc!=-1){
	if( (CkpvAccess(bgTraceCounter) % simParams->bgSplitNumProcs) == simParams->bgSplitMyProc){
	  // Do slow mambo simulation for this case!
	  // Counter is incremented only in startTraceBigSim()
	}
  }
//  fprintf(CkpvAccess(bgfp),"%s\n",params);


  char sequenceString[1024];
  sequenceString[0] = 0;
#ifdef CMK_BLUEGENE_CHARM
  sprintf();

  BgPrint("endTraceBigSim %f\n");
  // write event ID
  int seqno = tTIMELINEREC.length()-1;
  //  fprintf(CkpvAccess(bgfp),"%d ",seqno);
  sprintf(sequenceString, "seqno: { %d } ",seqno);

#endif
  fprintf(CkpvAccess(bgfp),"%s\n",params);

  
#ifdef CMK_BLUEGENE_CHARM
  printf(CkpvAccess(bgfp), "TRACEBIGSIM: event:{ %s } %s %s %s %s\n", eventname, sequenceString, timeString, perfCountString, paramString);
#else
  printf("TRACEBIGSIM: event:{ %s } %s %s %s %s\n", eventname, sequenceString, timeString, perfCountString, paramString);  
#endif


}




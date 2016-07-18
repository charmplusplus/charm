
// CMK_BIGSIM_CHARM is used during the emulation phase, 
// where you genereate the bgTrace logs and the parameter files

#if CMK_BIGSIM_CHARM
#include "blue.h"
#include "blue_impl.h"
#endif


//CkpvExtern(FILE *, bgfp);

extern unsigned long bgTraceCounter;
//extern bool insideTraceBracket;

extern "C" {

// Just some function declarations:
void initBigSimTrace(int outputParams, int outtiming);
void finalizeBigSimTrace();
void startTraceBigSim();

}

/** 
	startTraceBigSim() begins tracing an event with a specified set of parameters. These are convenience
	aliases so that a user can easily add/remove parameters while testing their application.
	Up to 20 parameters can be specified. These just call through to startTraceBigSim_20param().
*/
//void endTraceBigSim( char * eventName );
void endTraceBigSim( char * eventName , int stepno, double p1 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 , double p16 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 , double p16 , double p17 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 , double p16 , double p17 , double p18 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 , double p16 , double p17 , double p18 , double p19 );
void endTraceBigSim( char * eventName , int stepno, double p1 , double p2 , double p3 , double p4 , double p5 , double p6 , double p7 , double p8 , double p9 , double p10 , double p11 , double p12 , double p13 , double p14 , double p15 , double p16 , double p17 , double p18 , double p19 , double p20 );



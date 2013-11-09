#ifndef __AUTOPERFAPI__H__
#define __AUTOPERFAPI__H__

//three types of applications to start analysis
//global barrier for each time step
void autoPerfGlobalNextStep( );

//local time step
void autoPerfLocalNextStep( );

//no timestep, analysis starts when idle
void startAnalysisonIdle();
void autoPerfReset();

void registerAutoPerfDone(CkCallback cb, bool frameworkShouldAdvancePhase);

#endif

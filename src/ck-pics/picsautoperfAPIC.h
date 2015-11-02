#ifndef __AUTOPERFAPIC__H__
#define __AUTOPERFAPIC__H__

//three types of applications to start analysis
//global barrier for each time step

//local time step

//no timestep, analysis starts when idle
void startAnalysisonIdle();
void autoPerfReset();

void registerTuneGoal(int goalIndex);

void setUserDefinedGoal(double value);

#endif

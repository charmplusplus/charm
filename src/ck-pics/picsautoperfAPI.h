#ifndef __AUTOPERFAPI__H__
#define __AUTOPERFAPI__H__

#include "picsdefs.h"
#include "charm++.h"
#ifdef __cplusplus 
    extern "C" {
#endif

void PICS_registerAutoPerfDone(CkCallback cb, int frameworkShouldAdvancePhase);
void PICS_setNumOfPhases(bool fromGlobal, int num, char *names[]);

void PICS_startStep(bool fromGlobal);
void PICS_endStep(bool fromGlobal);
void PICS_endStepInc(bool fromGlobal, int incSteps);
void PICS_endStepResumeCb(bool fromGlobal,  CkCallback cb);
void PICS_startPhase(bool fromGlobal, int phaseId);
void PICS_endPhase(bool fromGlobal);

void PICS_localAutoPerfRun( );

void PICS_autoPerfRun();
void PICS_autoPerfRunResumeCb(CkCallback cb);

void PICS_SetAutoTimer();

void PICS_setCollectionMode(int m) ;

void PICS_setEvaluationMode(int m) ;

void PICS_markLDBStart(int appStep);

void PICS_markLDBEnd();

void PICS_setWarmUpSteps(int steps);

void PICS_setPauseSteps(int steps);

#ifdef __cplusplus 
    }
#endif
#endif

/*
 * =====================================================================================
 *
 *       Filename:  autoTuneAPI.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07/21/2013 09:10:57 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef _AUTOTUNERAPI_H
#define _AUTOTUNERAPI_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "picsdefs.h"

enum TP_DATATYPE{TP_INT, TP_DOUBLE, TP_FLOAT };
enum TP_MOVE_OP { OP_ADD, OP_MUL };
enum TP_STRATEGY {TS_EXHAUSTIVE, TS_SIMPLE, TS_LINEAR, TS_EXPONENTIAL, TS_RANDOM, TS_BISEARCH , PERF_LINEAR, PERF_QUADRIC, PERF_CUBIC, PERF_QUARTIC, PERF_DIRECT, TS_PERF_GUIDE};
enum PERFGOALS {NoTune, BestTimeStep, BestUtilPercentage, BestEnergyEfficiency, UserDefinedGoal};
enum STEER_BASE { PERF_MODEL_STEER, PERF_ANALYSIS_STEER};

typedef struct ControllableParameter
{
    char    name[30];
    enum    TP_DATATYPE datatype;
    double  defaultValue;
    double  currentValue;
    double  minValue;
    double  maxValue;
    //constraints
    double  bestValue;
    double  moveUnit;
    int     moveOP;
    int     effect;
    int     effectDirection;
    int     strategy;	
    double  effectScale;
    int     objectID; 
    int     uniqueSets; //whether the values can be different on different procs
}ControllableParameter;

#ifdef __cplusplus 
    extern "C" {
#endif

void PICS_registerTunableParameter(ControllableParameter *tp);

void PICS_registerLocalTunableParameterFields(char *name, enum TP_DATATYPE dt, double dv, double minV, double  maxV, double mU, int eff, int effDir, int moveOP, int stp);

void PICS_registerTunableParameterFields(char *name, enum TP_DATATYPE dt, double dv, double minV, double  maxV, double mU, int eff, int effDir, int moveOP, int stp, int  uniqueSet);

void PICS_registerTunableParameterFieldsWithChare(char *name, enum TP_DATATYPE dt, double dv, double minV, double  maxV, double mU, int moveOP,  int eff, int effDir, int stp, int uniqueSet, int _id, char *chareName);

double PICS_getTunedParameter(const char *name, int *valid);

void PICS_link_cp_chare(const char *cpName, const char *chareName);

#ifdef __cplusplus 
    }
#endif

#endif

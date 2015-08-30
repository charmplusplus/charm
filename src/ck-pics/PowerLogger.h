#ifndef _POWER_LOGGER_H
#define _POWER_LOGGER_H
//#include "PowerLog.decl.h"
class PowerLogger : public CBase_PowerLogger {
	public: 
	int coresPerProc;
	int numPEs;
	FILE *logFD;
  float stcpu,stcore,oldcpu,oldcore,divisor,stmem,totpower,oldmem,stuncore,olduncore,oldtotpower;
  double oldtime,STARTT,sttime;
	double staperf,oldaperf;
  double stmperf,oldmperf;
	
	void getPower(float *cpuP, float *coreP, float *memP, float *tempt, float *totpower);
	static void printPower(void *tt, double pp);
	PowerLogger(int num);
};

#endif

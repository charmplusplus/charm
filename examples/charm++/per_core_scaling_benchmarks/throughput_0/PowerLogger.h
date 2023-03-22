#ifndef _POWER_LOGGER_H
#define _POWER_LOGGER_H

class PowerLogger : public CBase_PowerLogger {
	public: 
	int coresPerProc;
	int numPEs;
	FILE *logFD;
    double stcpu, divisor,mem_unit;
	float stcore,oldcpu,oldcore,stmem,totpower,oldmem,stuncore,olduncore,oldtotpower;
	double oldtime,STARTT,sttime;
	double staperf,oldaperf;
	double stmperf,oldmperf;
	double totEnergy;
	void getPower(double *cpuP, float *coreP, float *memP, float *tempt, float *totpower, int cpu);
	static void printPower(void *tt, double pp);
	PowerLogger(int num);
};

#endif

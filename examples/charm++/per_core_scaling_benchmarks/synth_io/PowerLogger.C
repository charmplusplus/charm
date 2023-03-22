#include<iostream>
#include<fstream>
using namespace std;
//#include "PowerLogger.decl.h"
#include "PowerLogger.h"
#define core_msr   0x639
#define cpu_msr    0x611
#define mem_msr   0x619
#define energy_unit_msr 0x606
#define ENERGY_UNIT_OFFSET      0x08
#define ENERGY_UNIT_MASK        0x1F00
#define MSR_IA32_MPERF          0x000000e7
#define MSR_IA32_APERF          0x000000e8
#define IA32_TIME_STAMP_COUNTER 0x00000010

//#define core_msr   0x639
//#define cpu_msr    0x90
//#define mem_msr   0xa0
//#define energy_unit_msr 0x606
//#define ENERGY_UNIT_OFFSET      0x08
//#define ENERGY_UNIT_MASK        0x1F00
//#define MSR_IA32_MPERF          0x000000e7
//#define MSR_IA32_APERF          0x000000e8
//#define IA32_TIME_STAMP_COUNTER 0x00000010
//


int pduMap[20];

static float getTemp(int cpu)
{
	char val[10];
	FILE *f;
	char path[300];
	sprintf(path,"/sys/devices/platform/coretemp.0/temp%d_input",cpu+1);
	f=fopen(path,"r");
	if(f==NULL) {
		printf("[%d] FILE OPEN ERROR in: %s\n",CkMyPe(),path);
	}
	else {
		char* tmp = fgets(val,10,f);
		fclose(f);
	}
	return atof(val)/1000;
}
void mperf_(double *msr_value1, int cpu)
{
	char path[100];
	sprintf(path, "/dev/cpu/%d/msr", cpu);
	uint64_t msr_value=-1;
	int retval = 0;
	
	int fd = open(path, O_RDONLY);
	if (fd >= 0)
		retval = pread(fd, &msr_value, sizeof(msr_value), MSR_IA32_MPERF);
	else printf("Can not read MSR id=%d\n",core_msr);
	close(fd);
	
	*msr_value1=(double) msr_value;
}
void aperf_(double *msr_value1, int cpu)
{
	char path[100];
	sprintf(path, "/dev/cpu/%d/msr", cpu);
	uint64_t msr_value=-1;
	int retval = 0;
	
	int fd = open(path, O_RDONLY);
	if (fd >= 0)
		retval = pread(fd, &msr_value, sizeof(msr_value), MSR_IA32_APERF);
	else printf("Can not read MSR id=%d\n",core_msr);
	close(fd);
	*msr_value1=(double) msr_value;
}
uint64_t rdmsr(uint32_t msr_id, int cpu)
{
	char path[100];
	sprintf(path, "/dev/cpu/%d/msr", cpu);
	uint64_t msr_value;
	int retval = 0;
	
	int fd = open(path, O_RDONLY);
	if (fd >= 0)
		retval = pread(fd, &msr_value, sizeof(msr_value), msr_id);
	else printf("[%d] Can not read MSR id=%d\n", CkMyPe(), msr_id);
	close(fd);
	
	return retval == sizeof(msr_value) ? msr_value : 0;
}
void corepower_(float *msr_value1, int cpu)
{
	char path[100];
	sprintf(path, "/dev/cpu/%d/msr", cpu);
	uint64_t msr_value=-1;
	int retval = 0;
	
	int fd = open(path, O_RDONLY);
	if (fd >= 0)
		retval = pread(fd, &msr_value, sizeof(msr_value), core_msr);
	else printf("[%d] Can not read MSR corepower_ id=%d\n", CkMyPe(), core_msr);
	close(fd);
	*msr_value1=(double) msr_value;
}
void mempower_(float *msr_value1, int cpu)
{
	char path[100];
	sprintf(path, "/dev/cpu/%d/msr", cpu);
	uint64_t msr_value=-1;
	int retval = 0;

	int fd = open(path, O_RDONLY);
	if (fd >= 0)
		retval = pread(fd, &msr_value, sizeof(msr_value), mem_msr);
	else printf("[%d] Can not read MSR mempower_ id=%d\n", CkMyPe(), mem_msr);
	close(fd);
	*msr_value1=(double) msr_value;
}
void mytime_(double *val)
{
	struct timeval end;
	long xx;
	*val =0.0;
	gettimeofday(&end, NULL);
	xx = end.tv_sec * 1000000 + end.tv_usec;
	*val = ((double)xx)/1000000.0;
}
void cpupower_(double *msr_value1, int cpu)
{
	char path[100];
	sprintf(path, "/dev/cpu/%d/msr", cpu);
	uint64_t msr_value=-1;
	int retval = 0;
	
	int fd = open(path, O_RDONLY);
	if (fd >= 0)
	  retval = pread(fd, &msr_value, sizeof(msr_value), cpu_msr);
	else printf("[%d] Can not read MSR id=%d\n",CkMyPe(),cpu_msr);
	close(fd);
	*msr_value1=(double) msr_value;
}
void getpowerunit_(double *val, double *mem_unit)
{
    *mem_unit = pow(0.5,(double)16);
	uint32_t value;
	uint64_t msr_output = rdmsr(energy_unit_msr, 0);
	value = (msr_output & ENERGY_UNIT_MASK) >> ENERGY_UNIT_OFFSET;
	unsigned int energy_unit_divisor = 1 << value;
	*val = (double) energy_unit_divisor;
}
void PowerLogger::getPower(double *cpuP, float *coreP, float *memP, float *tempt, float *mytotpower, int cpu)
{
	oldcpu = stcpu;
	oldtime = sttime;
  	oldcore = stcore;
  	oldmem = stmem;
  	oldtotpower = totpower;
  	oldaperf = staperf;
  	oldmperf = stmperf;

	cpupower_(&(stcpu), cpu);
	corepower_(&(stcore), cpu);
	mempower_(&(stmem), cpu);
	mytime_(&(sttime));
	mperf_(&(stmperf), cpu);
	aperf_(&(staperf), cpu);

	double telap= sttime - oldtime;
	float *temp = new float[coresPerProc];
	float avgt=0;
	for(int i=0; i<coresPerProc; i++) {
		temp[i] = getTemp(i);
		avgt +=temp[i];
	}
	*tempt = avgt/=coresPerProc;
//  	CkPrintf("[%d] Coming in powerlogger printPower t:%f c:%d cpup:%f corep:%f memp:%f temp[0]:%f temp[5]:%f\n",CkMyPe(),CmiWallTimer()
//			,p->coresPerProc,(p->stcpu-p->oldcpu)/(telap*p->divisor), (p->stcore-p->oldcore)/(telap*p->divisor), 
//			(p->stmem-p->oldmem)/(telap*p->divisor),temp[0],temp[5]);

	// print time, cpu, core, mem
	*cpuP = (stcpu - oldcpu)/(telap*divisor);
    *coreP = (stcore - oldcore)/(telap*divisor);
    *memP = (stmem - oldmem)/(telap*divisor);
    *mytotpower = totpower;
	//	fprintf(p->logFD, "%f %f %f %f %f %f %f ",CmiWallTimer(),(p->stcpu-p->oldcpu)/(telap*p->divisor),
	//		(p->stcore-p->oldcore)/(telap*p->divisor),(p->stmem-p->oldmem)/(telap*p->divisor),p->totpower,
	//		2.0*(p->staperf-p->oldaperf)/(p->stmperf-p->oldmperf),avgt );

	//	// print temps
	//	for(int i=0;i<p->coresPerProc;i++)
	//		fprintf(p->logFD," %f",temp[i]);
	//	fprintf(p->logFD,"\n");
}
void PowerLogger::printPower(void *tt, double pp)
{
	PowerLogger *p = static_cast<PowerLogger *>(tt);
	p->oldcpu = p->stcpu;
	//p->oldcore = p->stcore;
	p->oldmem = p->stmem;
	p->oldtotpower = p->totpower;
	p->oldaperf = p->staperf;
	p->oldmperf = p->stmperf;

	p->oldtime = p->sttime;
	mytime_(&(p->sttime));

	int coreIndex = CkMyNode()%p->coresPerProc;
	cpupower_(&(p->stcpu), coreIndex);
	//corepower_(&(p->stcore), coreIndex);
	mempower_(&(p->stmem), coreIndex);
	mperf_(&(p->stmperf), coreIndex);
	aperf_(&(p->staperf), coreIndex);

	double telap= p->sttime-p->oldtime;
	float *temp = new float[p->coresPerProc];
	float avgt=0;
	for(int i=0; i<p->coresPerProc; i++){
		temp[i] = getTemp(i);
		avgt +=temp[i];
	}
	avgt/=p->coresPerProc;

	p->totEnergy += ((p->stcpu-p->oldcpu) + (p->stcore-p->oldcore))/p->divisor + (p->stmem-p->oldmem)*p->mem_unit;

//  	CkPrintf("[%d] Coming in powerlogger printPower t:%f c:%d cpup:%f corep:%f memp:%f temp[0]:%f temp[5]:%f\n",CkMyPe(),CmiWallTimer()
//			,p->coresPerProc,(p->stcpu-p->oldcpu)/(telap*p->divisor), (p->stcore-p->oldcore)/(telap*p->divisor), 
//			(p->stmem-p->oldmem)/(telap*p->divisor),temp[0],temp[5]);

	// print time, cpu, core, mem

	fprintf(p->logFD, "%f %f %f %f %f %f ",telap,(p->stcpu-p->oldcpu)/(telap*p->divisor),
		((p->stmem-p->oldmem)*p->mem_unit)/(telap),
		(p->staperf-p->oldaperf)/(p->stmperf-p->oldmperf), p->totEnergy, avgt);


	// print temps
	for(int i=0;i<p->coresPerProc;i++)
		fprintf(p->logFD," %f",temp[i]);
	fprintf(p->logFD,"\n");
}

PowerLogger::PowerLogger(int num)
{
	coresPerProc = num;
	numPEs = CkNumPes();
	totEnergy = 0;
	//if(CkMyPe()%coresPerProc==0) {
		char logFile[100];
		char hostname[1024];
		hostname[1023] = '\0';
		gethostname(hostname, 1023);
		snprintf(logFile, sizeof(logFile), "temp_pow.log.%s.%d",hostname, CkMyPe());
    	if ((logFD = fopen(logFile, "a"))) {
  			fprintf(logFD, "Time, Cpu Power, Mem Power, Avg Freq, TotEnergy, Avg Temp, Temperature[]\n");
		} else {
    		CkAbort("Couldn't open temperature/frequency log file");
    	}
		CkPrintf("[%d] ------------------- PowerLogger initialized --------------\n",CkMyPe());
		getpowerunit_(&divisor, &mem_unit);
		int coreIndex = CkMyNode()%coresPerProc;
		cpupower_(&stcpu, coreIndex);
		//corepower_(&stcore, coreIndex);
		mempower_(&stmem, coreIndex);
		mperf_(&stmperf, coreIndex);
		aperf_(&staperf, coreIndex);
		mytime_(&sttime);
		STARTT = sttime;

		CcdCallOnConditionKeep(CcdPERIODIC_10ms,&printPower,this);
	//}
}

//#include "PowerLogger.def.h"

#include<iostream>

#include<fstream>
#include "PowerLogger.decl.h"
#include "PowerLogger.h"
using namespace std;
#define core_msr   0x639
#define cpu_msr    0x611
#define mem_msr   0x619
#define energy_unit_msr 0x606
#define ENERGY_UNIT_OFFSET      0x08
#define ENERGY_UNIT_MASK        0x1F00
#define MSR_IA32_MPERF          0x000000e7
#define MSR_IA32_APERF          0x000000e8
#define IA32_TIME_STAMP_COUNTER 0x00000010


#if POWER_AVAIL

int pduMap[20];

static float getTemp(int cpu)
{
        char val[10];

        FILE *f;
                char path[300];
                sprintf(path,"/sys/devices/platform/coretemp.%d/temp1_input",cpu);
                f=fopen(path,"r");
                if (f==NULL) {
                        printf("[%d] 777 FILE OPEN ERROR in temp :%s\n",CkMyPe(),path);
                }
        else

        {
          fgets(val,10,f);
          fclose(f);
        }
        return atof(val)/1000;
}

void gettotpower_php(float *cpu)
{
        char str[1000];

				double mypow[200];
        FILE *f;
                char path[300];
                sprintf(path,"/home/sarood1/newpower");
                f=fopen(path,"r");
                if (f==NULL) {
                        printf("[%d] 777 FILE OPEN ERROR in gettotpower :%s\n",CkMyPe(),path);
                }
        else

        {
          fgets(str,1000,f);
				  char * pch;
					int n=0;
//				  printf ("Splitting string \"%s\" into tokens:\n",str);
				  pch = strtok (str," ");
				  while (pch != NULL)
				  {
            mypow[n] = atof(pch);
//            CkPrintf("n:%d p:%f ",n,mypow[n]);

				    pch = strtok (NULL, " ");
						n++;
				  }
//					CkPrintf("\n");					
          fclose(f);
        }
        char hostname[1024];
        hostname[1023] = '\0';
        gethostname(hostname, 1023);
				char * pch;
				char num[3];
			  pch = strstr (hostname,"tarekc");
				memcpy(num,pch+6,2);
				num[2]='\0';
//        *cpu = mypow[CkMyPe()/6];
				int tarekcnum = atoi(num);        	
				*cpu = mypow[tarekcnum-41];        
//				CkPrintf("host:%s num:%s id:%d pow:%f\n",hostname,num,tarekcnum-41,mypow[tarekcnum-41]);

}

void gettotpower_snmp(float *cpu)
{
	char syscmd[1000];
	int outlet = CkMyPe()/6;
	*cpu =0;
	sprintf(syscmd,"snmpwalk -v2c -cPower_Table tarek4234-ckt23-pdu.cs.illinois.edu .1.3.6.1.4.1.10418.17.2.5.5.1.60.1.1.%d | cut -d\" \" -f 4",pduMap[outlet]);
	FILE *fp = popen(syscmd,"r");
	char buf[1024];
	while (fgets(buf, 1024, fp)) {
  /* do something with buf */
		*cpu = atof(buf)/10.0;
//		CkPrintf("llll pdu:%d p:%f\n",pduMap[outlet],*cpu);
	}
}

void mperf_(double *msr_value1)
{

  char path[100];
  sprintf(path, "/dev/cpu/%d/msr", 0);

  uint64_t msr_value=-1;
  int retval = 0;

  int fd = open(path, O_RDONLY);
  if (fd >= 0)
    retval = pread(fd, &msr_value, sizeof(msr_value), MSR_IA32_MPERF);
  else printf("Can not read MSR id=%d\n",core_msr);
  close(fd);

 *msr_value1=(double) msr_value;
}

void aperf_(double *msr_value1)
{

  char path[100];
  sprintf(path, "/dev/cpu/%d/msr", 0);

  uint64_t msr_value=-1;
  int retval = 0;

  int fd = open(path, O_RDONLY);
  if (fd >= 0)
    retval = pread(fd, &msr_value, sizeof(msr_value), MSR_IA32_APERF);
  else printf("Can not read MSR id=%d\n",core_msr);
  close(fd);

 *msr_value1=(double) msr_value;
}


void gettotpower(float *cpu)
{
        char val[10];

        FILE *f;
                char path[300];
                sprintf(path,"/home/sarood1/LinuxBK/PowerMeter/power");
                f=fopen(path,"r");
                if (f==NULL) {
                        printf("[%d] 777 FILE OPEN ERROR in gettotpower :%s\n",CkMyPe(),path);
                }
        else

        {
          fgets(val,10,f);
          fclose(f);
        }
        *cpu = atof(val);
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
  else printf("[%d] Can not read MSR id=%d\n",CkMyPe(),msr_id);
  close(fd);

  return retval == sizeof(msr_value) ? msr_value : 0;
}

void corepower_(float *msr_value1)
{

  char path[100];
  sprintf(path, "/dev/cpu/%d/msr", 0);

  uint64_t msr_value=-1;
  int retval = 0;

  int fd = open(path, O_RDONLY);
  if (fd >= 0)
    retval = pread(fd, &msr_value, sizeof(msr_value), core_msr);
  else printf("[%d] Can not read MSR id=%d\n",CkMyPe(),core_msr);
  close(fd);

 *msr_value1=(double) msr_value;
}

void mempower_(float *msr_value1)
{

  char path[100];
  sprintf(path, "/dev/cpu/%d/msr", 0);

  uint64_t msr_value=-1;
  int retval = 0;

  int fd = open(path, O_RDONLY);
  if (fd >= 0)
    retval = pread(fd, &msr_value, sizeof(msr_value), mem_msr);
  else printf("[%d] Can not read MSR id=%d\n",CkMyPe(),mem_msr);
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

void cpupower_(float *msr_value1)
{

  char path[100];
  sprintf(path, "/dev/cpu/%d/msr", 0);

  uint64_t msr_value=-1;
  int retval = 0;

  int fd = open(path, O_RDONLY);
  if (fd >= 0)
    retval = pread(fd, &msr_value, sizeof(msr_value), cpu_msr);
  else printf("[%d] Can not read MSR id=%d\n",CkMyPe(),cpu_msr);
  close(fd);

 *msr_value1=(double) msr_value;
}

void getpowerunit_(float *val)
{
                uint32_t value;
                uint64_t msr_output = rdmsr(energy_unit_msr, 0);
                value = (msr_output & ENERGY_UNIT_MASK) >> ENERGY_UNIT_OFFSET;
                unsigned int energy_unit_divisor = 1 << value;
    *val = (double) energy_unit_divisor;
}

void PowerLogger::getPower(float *cpuP, float *coreP, float *memP, float *tempt, float *mytotpower)
{
	
		oldcpu = stcpu;
		oldtime = sttime;
  	oldcore = stcore;
  	oldmem = stmem;
  	oldtotpower = totpower;
  	oldaperf = staperf;
  	oldmperf = stmperf;

		cpupower_(&(stcpu));
		corepower_(&(stcore));
		mempower_(&(stmem));
		mytime_(&(sttime));
		mperf_(&(stmperf));
		aperf_(&(staperf));

		gettotpower_php(&(totpower));
		double telap= sttime - oldtime;
		float *temp = new float[coresPerProc];
		float avgt=0;
		for(int i=0;i<coresPerProc;i++)
		{
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
		p->oldtime = p->sttime;
  	p->oldcore = p->stcore;
  	p->oldmem = p->stmem;
  	p->oldtotpower = p->totpower;
  	p->oldaperf = p->staperf;
  	p->oldmperf = p->stmperf;

		cpupower_(&(p->stcpu));
		corepower_(&(p->stcore));
		mempower_(&(p->stmem));
		mytime_(&(p->sttime));
		mperf_(&(p->stmperf));
		aperf_(&(p->staperf));

		gettotpower_php(&(p->totpower));
		double telap= p->sttime-p->oldtime;
		float *temp = new float[p->coresPerProc];
		float avgt=0;
		for(int i=0;i<p->coresPerProc;i++)
		{
			temp[i] = getTemp(i);
			avgt +=temp[i];
		}
		avgt/=p->coresPerProc;
//  	CkPrintf("[%d] Coming in powerlogger printPower t:%f c:%d cpup:%f corep:%f memp:%f temp[0]:%f temp[5]:%f\n",CkMyPe(),CmiWallTimer()
//			,p->coresPerProc,(p->stcpu-p->oldcpu)/(telap*p->divisor), (p->stcore-p->oldcore)/(telap*p->divisor), 
//			(p->stmem-p->oldmem)/(telap*p->divisor),temp[0],temp[5]);

		// print time, cpu, core, mem
		fprintf(p->logFD, "%f %f %f %f %f %f %f ",CmiWallTimer(),(p->stcpu-p->oldcpu)/(telap*p->divisor),
			(p->stcore-p->oldcore)/(telap*p->divisor),(p->stmem-p->oldmem)/(telap*p->divisor),p->totpower,
			2.0*(p->staperf-p->oldaperf)/(p->stmperf-p->oldmperf),avgt );

		// print temps
		for(int i=0;i<p->coresPerProc;i++)
			fprintf(p->logFD," %f",temp[i]);
		fprintf(p->logFD,"\n");
	}

	PowerLogger::PowerLogger(int num)
	{
		pduMap[0] = 24; // tarekc41
		pduMap[1] = 23; // tarekc42
		pduMap[2] = 22; // tarekc43
		pduMap[3] = 21; // tarekc44
		pduMap[4] = 20; // tarekc45
		pduMap[5] = 19; // tarekc46
		pduMap[6] = 18; // tarekc47
		pduMap[7] = 16; // tarekc48
		pduMap[8] = 15; // tarekc49
		pduMap[9] = 14; // tarekc50
		pduMap[10] = 13; //tarekc51
		pduMap[11] = 12; //tarekc52
		pduMap[12] = 11; //tarekc53
		pduMap[13] = 10; //tarekc54
		pduMap[14] = 8; //tarekc55
		pduMap[15] = 7; //tarekc56
		pduMap[16] = 6; //tarekc57
		pduMap[17] = 5; //tarekc58
		pduMap[18] = 4; //tarekc59
		pduMap[19] = 3; //tarekc60
	
		//init_rapl();
		coresPerProc = 6;
		numPEs = CkNumPes();
		if(CkMyPe()%coresPerProc==0 )//&& CkMyPe()/coresPerProc!=2 && CkMyPe()/coresPerProc!=18)
		{
    	char logFile[100];
			char hostname[1024];
			hostname[1023] = '\0';
			gethostname(hostname, 1023);
  	  snprintf(logFile, sizeof(logFile), "temp_pow.log.%s.%d",hostname, CkMyPe());
	    if ((logFD = fopen(logFile, "a"))) {
      	fprintf(logFD, "Time, Cpu Power, Core Power, Mem Power, Tot Power, Avg Freq, Avg Temp, Temperature[]\n");
    	} else {
  	    CkAbort("Couldn't open temperature/frequency log file");
	    }

			CkPrintf("[%d] ------------------- PowerLogger initialized --------------\n",CkMyPe());
			//CcdCallOnConditionKeep(CcdPERIODIC_1s,&printPower,this);		
    getpowerunit_(&divisor);
    cpupower_(&stcpu);
    corepower_(&stcore);
    mempower_(&stmem);
    mperf_(&stmperf);
    aperf_(&staperf);

		gettotpower_php(&totpower);
    mytime_(&sttime);
    STARTT = sttime;

		}
	}

#else
void PowerLogger::getPower(float *cpuP, float *coreP, float *memP, float *tempt, float *mytotpower) {}
void PowerLogger::printPower(void *tt, double pp){}
PowerLogger::PowerLogger(int num){}
#endif
#include "PowerLogger.def.h"

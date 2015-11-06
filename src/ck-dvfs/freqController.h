/**
 * \addtogroup CkDvfs
*/
/*@{*/

#ifndef FREQCONTROLLER_H
#define FREQCONTROLLER_H

#include "charm.h"
#include "converse.h"
#include "middle.h"
#include "cklists.h"
#include "register.h" // for _entryTable

//the frequency settings are specific for Intel HASWELL architecture
//the settings with HASWELL may need to be changed for other architectures
#define HASWELL 1 

class FreqController {
  private:
	int num_cores;
	int num_avail_freqs;
	int *freqs;
  public:
	FreqController(){
#ifdef HASWELL
		num_cores = 4;
		num_avail_freqs = 16;
		freqs = new int[num_avail_freqs] {3501000, 3500000, 3300000, 3200000, 
										  3000000, 2800000, 2700000, 2500000, 
										  2300000, 2200000, 2000000, 1900000, 
										  1700000, 1500000, 1400000, 1200000};
#else
		//fill in for other architectues
#endif
	}
	
	//change the frequency of the processor to the specified freq level
	int changeFreq(int level){
		CkPrintf("Change freq to: %d\n", level);
		FILE *f;
		char path[300];
		sprintf(path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_setspeed",CkMyNode()%num_cores);
		f=fopen(path,"w");
		if (f==NULL) {
			printf("[%d] FILE OPEN ERROR: %s\n", CkMyNode(), path);
			return 0;
		} else {
			char write_freq[10];
			sprintf(write_freq, "%d", freqs[level]);
			fputs(write_freq,f);
			fclose(f);
			return 1;
		}
	}
	//change the frequency governor
	//options are: conservative ondemand userspace powersave performance 
	int changeGovernor(char* governor){
		FILE *f;
		char path[300];
		sprintf(path,"/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor",CkMyNode()%num_cores);
		f=fopen(path,"w");
		if (f==NULL) {
			printf("[%d] FILE OPEN ERROR: %s\n", CkMyNode(), path);
			return 0;
		} else {
			fputs(governor,f);
			fclose(f);
			return 1;
		}
	}
	//enable/disable turbo boost
	//0: diables, 1: enables
	int changeBoost(int enable){
		FILE *f;
		char path[300];
		sprintf(path,"/sys/devices/system/cpu/cpufreq/boost");
		f=fopen(path,"w");
		if (f==NULL) {
			printf("[%d] FILE OPEN ERROR: %s\n", CkMyNode(), path);
			return 0;
		} else {
			char option[10];
			sprintf(option, "%d", enable); 
			fputs(option,f);
			fclose(f);
			return 1;
		}
	}

};

CkpvExtern(FreqController*, _freqController);

void energyCharmInit(char **argv); //init energy module in ck

#endif /* FREQCONTROLLER_H */

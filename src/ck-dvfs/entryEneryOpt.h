/**
 * \addtogroup CkDvfs
*/
/*@{*/

#ifndef ENTRYENERGYOPT_H
#define ENTRYENERGYOPT_H

//#include "register.h" // for _entryTable
#include "charm.h"
#include "converse.h"
#include "entryEneryOpt.decl.h"

//the frequency settings are specific for Intel HASWELL architecture
//the settings with HASWELL may need to be changed for other architectures
#define HASWELL 1 

class EntryEnergyInfo{
  private:
	int id;
	double maxTime;
	double minTime;
	double avgTime;
	int callCount;
	bool optimize;
	int optimalFrequency;
  public:
	EntryEnergyInfo(): optimize(false), maxTime(0.0), minTime(0.0), callCount(0),
		optimalFrequency(0) {};

};

class ChareEntry {
  private:
	int num_entries;
	EntryEnergyInfo **entries;
  public:
	ChareEntry(){};
	~ChareEntry(){};

};

//collect entry method statistics for each chare
class ChareStats {
  private:
	ChareEntry **chareEntries;
  public:
	ChareStats(){};
	~ChareStats(){};

};

CkGroupID _energyOptimizer;

class EnergyOptMain : public Chare {
  public:
	EnergyOptMain(CkArgMsg *m){
		delete m;
		_energyOptimizer = CProxy_EnergyOptimizer::ckNew();
	};
	EnergyOptMain(CkMigrateMessage *m):Chare(m) {};
};

//EnergyOptimizer is responsible for collecting entry method statistics
//and applying frequency changes for optimal energy point for each entry method

class EnergyOptimizer : public CBase_EnergyOptimizer {
  private:
	int num_cores;
	int num_avail_freqs;
	int *freqs;
	ChareStats* energyStats;

  public:
	EnergyOptimizer(void){
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

		//userspace governor is needed to be able to change the frequency
		changeGovernor("userspace");
		//disable turbo-boost
		changeBoost(0);
		//set the frequency to be non-boost max level
		changeFreq(1);
	}

	EnergyOptimizer(CkMigrateMessage *m):CBase_EnergyOptimizer(m){};

	//change the frequency of the processor to the specified freq level
	int changeFreq(int level){
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

}; //end of EnergyOptimizer 


#endif /* ENTRYENERGYOPT_H */

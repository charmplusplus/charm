/**
 * \addtogroup CkDvfs
*/
/*@{*/

#ifndef FREQCONTROLLER_H
#define FREQCONTROLLER_H

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <fstream>

#include "charm.h"
#include "converse.h"
#include "middle.h"
#include "cklists.h"
#include "register.h" // for _entryTable

//the frequency settings are specific for Intel HASWELL architecture
//the settings with HASWELL may need to be changed for other architectures
#define HASWELL 1 
#define NUM_AVAIL_FREQS 16

#define CPU_MSR               0x611
#define MEM_MSR               0x619
#define ENERGY_UNIT_MSR       0x606
#define ENERGY_UNIT_OFFSET    0x08
#define ENERGY_UNIT_MASK      0x1F00

class FreqController {
  private:
    int num_cores;
    int *freqs;
    int cur_freq_level;

  public:
    FreqController(){
#if HASWELL
        num_cores = 4;
        freqs = new int[NUM_AVAIL_FREQS] {1200000, 1400000, 1500000, 1700000,
										  1900000, 2000000, 2200000, 2300000,
										  2500000, 2700000, 2800000, 3000000,
										  3200000, 3300000, 3500000, 3501000};
        cur_freq_level = -1;
#else
        //fill in for other architectues
#endif
    }

    //change the frequency of the processor to the specified freq level
    //return 1 on success, 0 on failure to change
    int changeFreq(int level){
        //check if current frequency is already at the desired level
        if(cur_freq_level == level) return 1;

        CkPrintf("[%d]: Change freq to level %d : %d\n", CkMyNode(), level, freqs[level]);
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
            cur_freq_level = level;
            return 1;
        }
    }
    int incrementFreqLevel(){
        if(cur_freq_level < NUM_AVAIL_FREQS-1)
            return changeFreq(cur_freq_level+1);
        else return 0;
    }
	int decrementFreqLevel(){
        if(cur_freq_level > 0)
            return changeFreq(cur_freq_level-1);
        else return 0;
	}
	int getCurFreqLevel(){ return cur_freq_level; }

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
    //read msr register
    uint64_t rdmsr(uint32_t msr_id, int cpu){
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
    //returns cpu power
    double cpuPower(){
        char path[100];
        sprintf(path, "/dev/cpu/%d/msr", CkMyNode());
        uint64_t msr_value=-1;
        int retval = 0;

        int fd = open(path, O_RDONLY);
        if (fd >= 0)
            retval = pread(fd, &msr_value, sizeof(msr_value), CPU_MSR);
        else printf("[%d] Can not read MSR id=%d\n", CkMyPe(), CPU_MSR);
        close(fd);
        return (double) msr_value;
    }
    //returns memory power
    double memPower(){
        char path[100];
        sprintf(path, "/dev/cpu/%d/msr", CkMyNode());
        uint64_t msr_value=-1;
        int retval = 0;

        int fd = open(path, O_RDONLY);
        if (fd >= 0)
            retval = pread(fd, &msr_value, sizeof(msr_value), MEM_MSR);
        else printf("[%d] Can not read MSR id=%d\n", CkMyPe(), MEM_MSR);
        close(fd);
        return (double) msr_value;
    }
    //power unit divisor
    double getPowUnit(){
        uint32_t value;
        uint64_t msr_output = rdmsr(ENERGY_UNIT_MSR, 0);
        value = (msr_output & ENERGY_UNIT_MASK) >> ENERGY_UNIT_OFFSET;
        unsigned int energy_unit_divisor = 1 << value;
        return (double) energy_unit_divisor;
    }
};

//CkpvExtern(FreqController*, _freqController);
void energyCharmInit(char **argv); //init energy module in ck

#endif /* FREQCONTROLLER_H */

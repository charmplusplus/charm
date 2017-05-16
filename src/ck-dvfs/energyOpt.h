/**
 * \addtogroup CkDvfs
*/
/*@{*/

#ifndef ENERGYOPT_H
#define ENERGYOPT_H

#include <vector>
#include "freqController.h"
#include "energyOpt.decl.h"

class EntryEnergyInfo;
class ChareEntry;
class ChareStats;
class EnergyOptimizer;

extern CkGroupID _energyOptimizer;

//repersents statisctics about an entry method in a chare
class EntryEnergyInfo{
  private:
    //int id;                             //epidx
    int freqCallCount[NUM_AVAIL_FREQS]; //how many time the entry is called for each freq level
    double freqTime[NUM_AVAIL_FREQS];   //avg time for each frequency level
    double freqPow[NUM_AVAIL_FREQS];    //power for each frequency level
    bool optimize;                      //whether to change the freq for this function or not
    int optimalFrequency;               //desired freq level for this entry
    int num_trials;                     //number of different frequency level trials
    int optFreqFound;                   //if the optimal frequency is found yet
  public:
    EntryEnergyInfo(): optimize(false), optimalFrequency(0), optFreqFound(0), num_trials(0) {
        for(int i=0; i<NUM_AVAIL_FREQS; ++i){
            freqCallCount[i]=0;
            freqTime[i]=0;
            freqPow[i]=0;
        }
    }
    //save information about the entry method
    void addEntryInfo(double duration, int freqLevel, double pow){
        num_trials++;
        CkAssert(freqLevel < NUM_AVAIL_FREQS && freqLevel >= 0);
        //CkPrintf("addEntryInfo. %d-%d\n", NUM_AVAIL_FREQS, freqLevel);
        int count = freqCallCount[freqLevel]; //previous info count
        freqPow[freqLevel] = (freqPow[freqLevel]*count+pow)/(count+1); //calculate the avg value with the past value
        freqTime[freqLevel] = (freqTime[freqLevel]*count+duration)/(count+1); // "
        freqCallCount[freqLevel] = count+1; //update call count
    }
    //find the energy minimal frequency
    //energy = power * time
    void calculateOptimalFreqLevel(){
        if(num_trials<=0) return;
        double minEnergy = -1;
        int optFreqLevel = -1;
        for(int l=0; l<NUM_AVAIL_FREQS; ++l){
            double energy = freqPow[l] * freqTime[l]; //what if it's 0?
            if(minEnergy == -1){
                minEnergy = energy;
                optFreqLevel = l;
            }
            else if(energy < minEnergy){
                optFreqLevel = l;
                minEnergy = energy;
            }
        }
        optimalFrequency = optFreqLevel;
        optFreqFound = 1;
    }
    void printOptimalFreqInfo(){
        if(num_trials<=0) return;
        for(int l=0; l<NUM_AVAIL_FREQS; ++l){
            CkPrintf("[%d] - %d - Pow:%f, Time:%f, Enrgy:%f" , CkMyNode(), l, freqPow[l], freqTime[l], freqPow[l] * freqTime[l]);
            if(optimalFrequency==l) CkPrintf(" - OPT\n");
            else CkPrintf("\n");
        }
    }
    int getOptimalFreqLevel(){
        return optimalFrequency;
    }
    double getAvgExecTime(int freqLevel){
        return freqTime[freqLevel];
    }
    int isOptimalFreqFound(){
        return optFreqFound;
    }

};

//collect entry method statistics for entry in each chare
class ChareEntry {
  public:
    int numEntries;
    std::vector<EntryEnergyInfo> entryStats;
    ChareEntry(int entries): numEntries(entries){
        entryStats.resize(numEntries);
    }
    ~ChareEntry(){}
    ChareEntry(const ChareEntry& ce): numEntries(ce.numEntries){
        entryStats.resize(numEntries);
    }
    void addChareEntryStat(double duration, int freqLevel, double pow, int epIdx){
        //CkPrintf("epIdx: %d, numEntries: %d\n", epIdx, numEntries);
        CkAssert(epIdx<numEntries && epIdx >= 0);
        entryStats[epIdx].addEntryInfo(duration, freqLevel, pow);
    }
    void calculateOptimalFreqLevel(){
        for(int i=0; i<numEntries; ++i){
            entryStats[i].calculateOptimalFreqLevel();
            //CkPrintf("[%d] Entry[%d], optimal freq level found: %d\n", CkMyPe(),
            //    i, entryStats[i].getOptimalFreqLevel());
            if(i==164) entryStats[i].printOptimalFreqInfo(); //BILGE
        }
    }
};

//collect entry method statistics for chares
class ChareStats {
  public:
    int numChares, numEntries;
    std::vector<ChareEntry> chareStats;
    ChareStats(int chares, int entries): numChares(chares), numEntries(entries){
        CkPrintf("[%d] ChareStats chares: %d, entries: %d\n", CkMyPe(), chares, entries);
        chareStats.resize(numChares, ChareEntry(numEntries));
    }
    ~ChareStats(){}
    ChareStats(const ChareStats& c): numChares(c.numChares), numEntries(c.numEntries){
        chareStats.resize(numChares, ChareEntry(numEntries));
    }
    void addChareStat(double duration, int freqLevel, double pow, int epIdx, int objIdx){
        //CkPrintf("objIdx: %d, numChares: %d\n", objIdx, numChares);
        CkAssert(objIdx<numChares && objIdx >= 0);
        chareStats[objIdx].addChareEntryStat(duration, freqLevel, pow, epIdx);
    }
    void calculateOptimalFreqLevel(){
        for(int i=0; i<numChares; i++)
            chareStats[i].calculateOptimalFreqLevel();
        CkPrintf("[%d] optimal freqs found...\n", CkMyPe());

    }
};

class EnergyOptMain : public Chare {
  public:
    EnergyOptMain(CkArgMsg *m){
        delete m;
        _energyOptimizer = CProxy_EnergyOptimizer::ckNew();
        CkPrintf("EnergyOptMain CONSTRUCTOR\n");
    };
    EnergyOptMain(CkMigrateMessage *m):Chare(m) {};
};

//EnergyOptimizer is responsible for collecting entry method statistics
//and applying frequency changes for optimal energy point for each entry method

#define TRIAL_TIMEOUT           5 //try every freq level for this many seconds
#define FREQ_CHANGE_LATENCY     0.00001 //seconds
#define ENTRY_THRESHOLD         FREQ_CHANGE_LATENCY*10

//Periodically log power
//void powerCounterAdd(void *EO, double curWallTime);

class EnergyOptimizer : public IrrGroup {

  private:
    ChareStats* energyStats;
    bool statsOn;
    double timer;

  public:
    FreqController* freqController;
    int powerCounter;
    int powerSum;
    EnergyOptimizer(void){
        //BILGE TODO: track stats of all funtions or only user functions?
        energyStats = new ChareStats(_chareTable.size(), _entryTable.size());
        freqController = new FreqController();
        CkPrintf("[%d] EnergyOptimizer created for %d chares and %d entries!\n", CkMyNode(), _chareTable.size(), _entryTable.size());

        //start collecting stats from the beginning, it'll be turned of once
        //settled on optimal frequency
        statsOn = true;

        //userspace governor is needed to be able to change the frequency
        freqController->changeGovernor("userspace");
        //disable turbo-boost
        freqController->changeBoost(0);
        //set the frequency to be max level
        freqController->changeFreq(15);

        //start the timer
        timer = CkWallTimer();

        //Read the power periodically
        powerCounter = 0;
        powerSum = 0;
        //CcdCallOnConditionKeep(CcdPERIODIC_100ms, &powerCounterAdd, this);
    }

    EnergyOptimizer(CkMigrateMessage *m){};
    ~EnergyOptimizer(){
        delete energyStats;
    }

    bool isStatsOn(){ return statsOn; }

    //reset the power counters
    void powerCounterReset(){ powerCounter = 0; powerSum = 0; }
    //caculate the average power based on the counter data
    double powerCounterGetAvgPow(){
        return powerSum / powerCounter;
    }

    //Periodically log power
    static void powerCounterAdd(void *EO, double curWallTime){
        EnergyOptimizer *eo = static_cast<EnergyOptimizer *>(EO);
        double curPow = eo->freqController->cpuPower() + eo->freqController->memPower();
        eo->powerCounter += 1;
        eo->powerSum += curPow;
    }

    void addEntryStat(double duration, int epIdx, int objIdx){
        CkPrintf("[%d] Done.. Adding entry stats!\n", CkMyNode());

        energyStats->addChareStat(duration, freqController->getCurFreqLevel(),
                powerCounterGetAvgPow(), epIdx, objIdx);
        if(statsOn){
            //is it time to move to the next freq level?
            double now = CkWallTimer();
            if(timer - now > TRIAL_TIMEOUT){
                if(!freqController->decrementFreqLevel()){
                    //stats collection done, calculate optimal frequency
                    statsOn = false;
                    energyStats->calculateOptimalFreqLevel();
                }
                timer = now;
            }
        }
    }
    void addEntryStat(double duration, double energy, int epIdx, int objIdx){
        //CkPrintf("[%d] Adding entry stats! epIdx: %d, objIdx: %d...\n", CkMyNode(), epIdx, objIdx);
        energyStats->addChareStat(duration, freqController->getCurFreqLevel(),
                energy/duration, epIdx, objIdx);
    }
    void adjustFrequency(int epIdx, int objIdx){
        CkPrintf("[%d] adjustFrequency!\n", CkMyNode());
        if(statsOn){
            //is it time to move to the next freq level?
            double now = CkWallTimer();
            CkPrintf("[%d] adjustFrequency: %f s!\n", now-timer );
            if(now-timer > TRIAL_TIMEOUT){
                if(!freqController->decrementFreqLevel()){
                    //stats collection done, calculate optimal frequency
                    CkPrintf("[%d] Stats collection done!\n", CkMyNode());
                    statsOn = false;
                    energyStats->calculateOptimalFreqLevel();
                }
                timer = now;
            }
        }else{
            //set the calculated optimal frequency
            //if the optimal freq is the same with current freq,
            //do not do anything
            int optFreqFound = energyStats->chareStats[objIdx].entryStats[epIdx].isOptimalFreqFound();
            if(optFreqFound){
                CkPrintf("[%d] OPT freq found.\n", CkMyNode());
                int optFreq = energyStats->chareStats[objIdx].entryStats[epIdx].getOptimalFreqLevel();
                int curFreq = freqController->getCurFreqLevel();
                double duration = energyStats->chareStats[objIdx].entryStats[epIdx].getAvgExecTime(curFreq);
                if(optFreq != curFreq && duration > ENTRY_THRESHOLD){
                    CkPrintf("[%d] Changing to OPT freq: %d.\n", CkMyNode(), optFreq);
                    freqController->changeFreq(optFreq);
                }
            }
        }
    }

}; //end of EnergyOptimizer

#endif /* ENERGYOPT_H */

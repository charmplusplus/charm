/**
 * \addtogroup CkDvfs
*/
/*@{*/

#ifndef ENERGYOPT_H
#define ENERGYOPT_H

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
    EntryEnergyInfo(): optimize(false), optimalFrequency(0), optFreqFound(0) {};
    void addEntryInfo(int duration, int freqLevel, double pow){
        freqCallCount[freqLevel]++;
        num_trials++;
        freqPow[freqLevel] = pow;
        freqTime[freqLevel] = duration;
    }
    //find the energy minimal frequency
    //energy = power * time
    void calculateOptimalFreqLevel(){
        double minEnergy = -1;
        int optFreqLevel = -1;
        for(int l=0; l<NUM_AVAIL_FREQS; l++){
            int energy = freqPow[l] * freqTime[l];
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

};

//collect entry method statistics for entry in each chare
class ChareEntry {
  private:
    int numEntries;
    EntryEnergyInfo **entryStats;
  public:
    ChareEntry(int entries): numEntries(entries){
        entryStats = new EntryEnergyInfo*[numEntries];
        for(int i=0; i<numEntries; i++)
            entryStats[i] = new EntryEnergyInfo();
    };
    ~ChareEntry(){
        delete entryStats;
    };
    void addChareEntryStat(int duration, int freqLevel, double pow, int epIdx){
        entryStats[epIdx]->addEntryInfo(duration, freqLevel, pow);
    }
    void calculateOptimalFreqLevel(){
        for(int i=0; i<numEntries; i++)
            entryStats[i]->calculateOptimalFreqLevel();
    }
};

//collect entry method statistics for each chare
class ChareStats {
  private:
    int numChares, numEntries;
    ChareEntry **chareStats;
  public:
    ChareStats(int chares, int entries): numChares(chares), numEntries(entries){
        chareStats = new ChareEntry*[numChares];
        for(int i=0; i<numChares; i++)
            chareStats[i] = new ChareEntry(numEntries);
    };
    ~ChareStats(){
        for(int i=0; i<numChares; i++)
            delete chareStats[i];
        delete chareStats;
    };
    void addChareStat(double duration, int freqLevel, double pow, int epIdx, int objIdx){
        chareStats[objIdx]->addChareEntryStat(duration, freqLevel, pow, epIdx);
    }
    void calculateOptimalFreqLevel(){
        for(int i=0; i<numChares; i++)
            chareStats[i]->calculateOptimalFreqLevel();
    }
};

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

#define TRIAL_TIMEOUT   30 //try every freq level for this many seconds

//Periodically log power
void powerCounterAdd(void *EO, double curWallTime);

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
        //set the frequency to be non-boost max level
        freqController->changeFreq(1);

        //start the timer
        timer = CkWallTimer();

        //Read the power periodically
        powerCounter = 0;
        powerSum = 0;
        CcdCallOnConditionKeep(CcdPERIODIC_100ms, &powerCounterAdd, this);
        //CcdCancelCallOnConditionKeep(int condnum, int idx)

    }

    EnergyOptimizer(CkMigrateMessage *m){};

    bool isStatsOn(){ return statsOn; }

    //reset the power counters
    void powerCounterReset(){ powerCounter = 0; powerSum = 0; }
    //caculate the average power based on the counter data
    double powerCounterGetAvgPow(){
        return powerSum / powerCounter;
    }

    void addEntryStat(double duration, int epIdx, int objIdx){
        CkPrintf("[%d] Adding entry stats!\n", CkMyNode());

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
    };

}; //end of EnergyOptimizer

//Periodically log power
void powerCounterAdd(void *EO, double curWallTime){
    EnergyOptimizer *eo = static_cast<EnergyOptimizer *>(EO);
    double curPow = eo->freqController->cpuPower() + eo->freqController->memPower();
    eo->powerCounter += 1;
    eo->powerSum += curPow;
}

#endif /* ENERGYOPT_H */

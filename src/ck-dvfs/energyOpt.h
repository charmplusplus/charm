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
    int id;                             //epidx
    int freqCallCount[NUM_AVAIL_FREQS]; //how many time the entry is called for each freq level
    double freqTime[NUM_AVAIL_FREQS];   //avg time for each frequency level
    double freqPow[NUM_AVAIL_FREQS];    //power for each frequency level
    bool optimize;                      //whether to change the freq for this function or not
    int optimalFrequency;               //desired freq level for this entry
    int num_trials;                     //number of different frequency level trials
    int optFreqFound;                   //if the optimal frequency is found yet
  public:
    EntryEnergyInfo(): optimize(false), optimalFrequency(0) {};
    void addEntryInfo(){
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

#define TRIAL_TIMEOUT   60 //try every freq level for this many seconds

class EnergyOptimizer : public IrrGroup {

  private:
    ChareStats* energyStats;
    FreqController* freqController;
    bool statsOn;
    double timer;

  public:
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
    }

    EnergyOptimizer(CkMigrateMessage *m){};

    bool isStatsOn(){ return statsOn; }

    void addEntryStat(double time, int epIdx, int objIdx){
        CkPrintf("[%d] Adding entry stats!\n", CkMyNode());

        //is it time to move to the next freq level?
        double now = CkWallTimer();
        if(timer - now > TRIAL_TIMEOUT){
            if(!freqController->incrementFreqLevel()){
                //stats collection done, calculate optimal frequency
                statsOn = false;
            }
            timer = now;

        }
    };

}; //end of EnergyOptimizer

#endif /* ENERGYOPT_H */

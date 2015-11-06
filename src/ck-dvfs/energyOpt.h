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

//CkGroupID _energyOptimizer;
//CProxy_EnergyOptimizer _energyOptimizer;

//repersents statisctics about an entry method in a chare
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
		CkGroupID _energyOptimizer = CProxy_EnergyOptimizer::ckNew();
	};
	EnergyOptMain(CkMigrateMessage *m):Chare(m) {};
};

//EnergyOptimizer is responsible for collecting entry method statistics
//and applying frequency changes for optimal energy point for each entry method

class EnergyOptimizer : public CBase_EnergyOptimizer {

  private:
	ChareStats* energyStats;

  public:
	EnergyOptimizer(void){
		CkPrintf("[%d] EnergyOptimizer created!\n", CkMyNode());

		energyStats = new ChareStats(_chareTable.size(), _entryTable.size());

		//userspace governor is needed to be able to change the frequency
		//CkpvAccess(_freqController)->changeGovernor("userspace");
		//disable turbo-boost
		//CkpvAccess(_freqController)->changeBoost(0);
		//set the frequency to be non-boost max level
		//CkpvAccess(_freqController)->changeFreq(1);
	}

	EnergyOptimizer(CkMigrateMessage *m):CBase_EnergyOptimizer(m){};

}; //end of EnergyOptimizer 

#endif /* ENERGYOPT_H */

/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef CENTRALLB_H
#define CENTRALLB_H

#include <LBDatabase.h>
#include "CentralLB.decl.h"

extern CkGroupID loadbalancer;

void CreateCentralLB();

class CLBStatsMsg;
class CLBMigrateMsg;

class CentralLB : public Group
{
public:
  CentralLB();
  ~CentralLB();
  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier

  void ReceiveStats(CLBStatsMsg *); 		// Receive stats on PE 0
  void ResumeClients(void);                     // Resuming clients needs
	                                        // to be resumed via message
  void ReceiveMigration(CLBMigrateMsg *); 	// Receive migration data

  // Migrated-element callback
  static void staticMigrated(void* me, LDObjHandle h);
  void Migrated(LDObjHandle h);

  void MigrationDone(void);  // Call when migration is complete
  int step() { return mystep; };

  struct MigrateInfo {  // Used in CLBMigrateMsg
    LDObjHandle obj;
    int from_pe;
    int to_pe;
  };

  struct LDStats {  // Passed to Strategy
    double total_walltime;
    double total_cputime;
    double idletime;
    double bg_walltime;
    double bg_cputime;
    int pe_speed;
    double utilization;
    CmiBool available;
    
    int n_objs;
    LDObjData* objData;
    int n_comm;
    LDCommData* commData;
  };

   CLBMigrateMsg* callStrategy(LDStats* stats,int count){
	return Strategy(stats,count);
   };

protected:
  virtual CmiBool QueryBalanceNow(int) { return CmiTrue; };  
  virtual CLBMigrateMsg* Strategy(LDStats* stats,int count);
  LBDatabase* theLbdb;

private:  
  int mystep;
  int myspeed;
  int stats_msg_count;
  CLBStatsMsg** statsMsgsList;
  LDStats* statsDataList;
  int migrates_completed;
  int migrates_expected;
  double start_lb_time;
};

class CLBStatsMsg : public CMessage_CLBStatsMsg {
public:
  int from_pe;
  int serial;
  int pe_speed;
  double total_walltime;
  double total_cputime;
  double idletime;
  double bg_walltime;
  double bg_cputime;
  int n_objs;
  LDObjData *objData;
  int n_comm;
  LDCommData *commData;

  // Other methods & data members 

  static void* alloc(int msgnum, size_t size, int* array, int priobits); 
  static void* pack(CLBStatsMsg* in); 
  static CLBStatsMsg* unpack(void* in); 
}; 

class CLBMigrateMsg : public CMessage_CLBMigrateMsg {
public:
  int n_moves;
  CentralLB::MigrateInfo* moves;

  // Other methods & data members 

  static void* alloc(int msgnum, size_t size, int* array, int priobits); 
  static void* pack(CLBMigrateMsg* in); 
  static CLBMigrateMsg* unpack(void* in); 
}; 

#endif /* CENTRALLB_H */

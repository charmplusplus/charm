#ifndef CENTRALLB_H
#define CENTRALLB_H

#include <LBDatabase.h>
#include "CentralLB.decl.h"


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
  void ReceiveMigration(CLBMigrateMsg *); 	// Receive migration data

  // Migrated-element callback
  static void staticMigrated(void* me, LDObjHandle h);
  void Migrated(LDObjHandle h);

  void MigrationDone(void);  // Call when migration is complete

  struct MigrateInfo {  // Used in CLBMigrateMsg
    LDObjHandle obj;
    int from_pe;
    int to_pe;
  };

  struct LDStats {  // Passed to Strategy
    int n_objs;
    LDObjData* objData;
    int n_comm;
    LDCommData* commData;
  };

protected:
  virtual CmiBool QueryBalanceNow(int) { return CmiTrue; };  
  virtual CLBMigrateMsg* Strategy(LDStats* stats,int count);

private:  
  int step;
  LBDatabase* theLbdb;
  int stats_msg_count;
  CLBStatsMsg** statsMsgsList;
  LDStats* statsDataList;
  int migrates_completed;
  int migrates_expected;
};

class CLBStatsMsg : public CMessage_CLBStatsMsg {
public:
  int from_pe;
  int serial;
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

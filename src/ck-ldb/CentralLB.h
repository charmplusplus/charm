/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef CENTRALLB_H
#define CENTRALLB_H

#include "BaseLB.h"
#include "CentralLB.decl.h"
#include "SimResults.h"

extern CkGroupID loadbalancer;

#define PER_MESSAGE_RECV_OVERHEAD 0.01
#define PER_MESSAGE_SEND_OVERHEAD 0.01
#define PER_BYTE_RECV_OVERHEAD 0.0001
#define PER_BYTE_SEND_OVERHEAD 0.0001

void CreateCentralLB();
void set_avail_vector(char * bitmap);

class CLBStatsMsg;
class CLBSimResults;

/// for backward compatibility
typedef LBMigrateMsg  CLBMigrateMsg;

class CentralLB : public CBase_CentralLB
{
public:
  CentralLB();
  ~CentralLB();
  CentralLB(CkMigrateMessage *m) {}
  static void staticAtSync(void*);
  void AtSync(void); // Everything is at the PE barrier
  void ProcessAtSync(void); // Receive a message from AtSync to avoid
                            // making projections output look funny

  void ReceiveStats(CLBStatsMsg *); 		// Receive stats on PE 0
  void ResumeClients(void);                     // Resuming clients needs
	                                        // to be resumed via message
  void ReceiveMigration(LBMigrateMsg *); 	// Receive migration data

  // manuall start load balancing
  inline void StartLB() { thisProxy.ProcessAtSync(); }
  static void staticStartLB(void* data);

  // Migrated-element callback
  static void staticMigrated(void* me, LDObjHandle h);
  void Migrated(LDObjHandle h);

  void MigrationDone(int balancing);  // Call when migration is complete
  int step() { return mystep; };

  void set_avail_vector(char *new_vector);

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

   LBMigrateMsg* callStrategy(LDStats* stats,int count){
	return Strategy(stats,count);
   };

  int cur_ld_balancer;
  char *avail_vector;
  /* for Node 0 */
  int new_ld_balancer;

  void readStatsMsgs(const char* filename);
  void writeStatsMsgs(const char* filename);

protected:
  virtual CmiBool QueryBalanceNow(int) { return CmiTrue; };  
  virtual CmiBool QueryDumpData() { return CmiFalse; };  
  virtual LBMigrateMsg* Strategy(LDStats* stats,int count);

  void simulation();
  void FindSimResults(LDStats* stats, int count, LBMigrateMsg* msg, CLBSimResults* simResults);
  void RemoveNonMigratable(LDStats* statsDataList, int count);

private:  

  int mystep;
  int myspeed;
  int stats_msg_count;
  CLBStatsMsg** statsMsgsList;
  LDStats* statsDataList;
  int migrates_completed;
  int migrates_expected;
  double start_lb_time;

public:
  int useMem();
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

  char * avail_vector;
  int next_lb;
}; 

#endif /* CENTRALLB_H */

/*@}*/



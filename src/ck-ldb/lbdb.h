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

#ifndef LBDBH_H
#define LBDBH_H

#include <converse.h>
#include <charm.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* cvoid; /* To eliminate warnings, because a C void* is not
			the same as a C++ void* */

  /*  User-defined object IDs will be 4 ints long */
#define OBJ_ID_SZ 4

#if CMK_LBDB_ON
typedef struct {
  void *handle;
} LDHandle;
#else
typedef int LDHandle;
#endif

typedef struct {
  CkGroupID id;
} LDOMid;

typedef struct {
  LDHandle ldb;
  void *user_ptr;
  LDOMid id;
  int handle;
} LDOMHandle;

typedef struct _LDObjid {
  int id[OBJ_ID_SZ];
#ifdef __cplusplus
  CmiBool operator==(const struct _LDObjid& objid) const {
    for (int i=0; i<OBJ_ID_SZ; i++) if (id[i] != objid.id[i]) return CmiFalse;
    return CmiTrue;
  }
#endif
} LDObjid;

typedef struct {
  LDOMHandle omhandle;
  void *user_ptr;
  LDObjid id;
  int handle;
} LDObjHandle;

typedef struct {
  LDObjHandle handle;
  LDOMHandle omHandle;
  double cpuTime;
  double wallTime;
  CmiBool migratable;
#ifdef __cplusplus
  inline const LDOMid &omID() const { return omHandle.id; }
  inline const LDObjid &id() const { return handle.id; }
#endif
} LDObjData;

typedef struct {
  int src_proc;
  LDOMid senderOM;
  LDObjid sender;
  int dest_proc;
  LDOMid receiverOM;
  LDObjid receiver;
  int messages;
  int bytes;
#ifdef __cplusplus
  inline int from_proc() const { return (src_proc != -1); }
  inline int to_proc() const { return (dest_proc != -1); }
#endif
} LDCommData;

/*
 * Requests to load balancer
 */
void LBBalance(void *param);
void LBCollectStatsOn(void);
void LBCollectStatsOff(void);

/*
 * Callbacks from database to object managers
 */
typedef void (*LDMigrateFn)(LDObjHandle handle, int dest);
typedef void (*LDStatsFn)(LDOMHandle h, int state);
typedef void (*LDQueryEstLoadFn)(LDOMHandle h);

typedef struct {
  LDMigrateFn migrate;
  LDStatsFn setStats;
  LDQueryEstLoadFn queryEstLoad;
} LDCallbacks;

/*
 * Calls from object managers to load database
 */
#if CMK_LBDB_ON
LDHandle LDCreate(void);
#else
#define LDCreate() 0
#endif

LDOMHandle LDRegisterOM(LDHandle lbdb, LDOMid userID, 
			void *userptr, LDCallbacks cb);
void LDRegisteringObjects(LDOMHandle _h);
void LDDoneRegisteringObjects(LDOMHandle _h);

LDObjHandle LDRegisterObj(LDOMHandle h, LDObjid id, void *userptr,
			  int migratable);
void LDUnregisterObj(LDObjHandle h);

void LDObjTime(LDObjHandle h, double walltime, double cputime);
int LDRunningObject(LDHandle _h, LDObjHandle* _o );
void LDObjectStart(const LDObjHandle &_h);
void LDObjectStop(const LDObjHandle &_h);
void LDSend(const LDOMHandle &destOM, const LDObjid &destid, unsigned int bytes);

void LDMessage(LDObjHandle from, 
	       LDOMid toOM, LDObjid *toID, int bytes);

void LDEstObjLoad(LDObjHandle h, double load);
void LDNonMigratable(LDObjHandle h);
void LDMigratable(LDObjHandle h);
void LDDumpDatabase(LDHandle lbdb);

/*
 * Calls from load balancer to load database
 */  
typedef void (*LDMigratedFn)(void* data, LDObjHandle handle);
void LDNotifyMigrated(LDHandle lbdb, LDMigratedFn fn, void* data);

void LDCollectStatsOn(LDHandle lbdb);
void LDCollectStatsOff(LDHandle lbdb);
void LDQueryEstLoad(LDHandle bdb);

int LDGetObjDataSz(LDHandle lbdb);
void LDGetObjData(LDHandle lbdb, LDObjData *data);

int LDGetCommDataSz(LDHandle lbdb);
void LDGetCommData(LDHandle lbdb, LDCommData *data);

void LDBackgroundLoad(LDHandle lbdb, double *walltime, double *cputime);
void LDIdleTime(LDHandle lbdb, double *walltime);
void LDTotalTime(LDHandle lbdb, double *walltime, double *cputime);

void LDClearLoads(LDHandle lbdb);
void LDMigrate(LDObjHandle h, int dest);
void LDMigrated(LDObjHandle h);

/*
 * Local Barrier calls
 */
typedef void (*LDBarrierFn)(void *user_ptr);
typedef void (*LDResumeFn)(void *user_ptr);

typedef struct {
  int serial;
} LDBarrierClient;

typedef struct {
  int serial;
} LDBarrierReceiver;

LDBarrierClient LDAddLocalBarrierClient(LDHandle lbdb,LDResumeFn fn,
					void* data);
void LDRemoveLocalBarrierClient(LDHandle lbdb, LDBarrierClient h);
LDBarrierReceiver LDAddLocalBarrierReceiver(LDHandle lbdb,LDBarrierFn fn,
					    void* data);
void LDRemoveLocalBarrierReceiver(LDHandle lbdb,LDBarrierReceiver h);
void LDAtLocalBarrier(LDHandle lbdb, LDBarrierClient h);
void LDResumeClients(LDHandle lbdb);
int LDProcessorSpeed();
CmiBool LDOMidEqual(const LDOMid &i1, const LDOMid &i2);
CmiBool LDObjIDEqual(const LDObjid &i1, const LDObjid &i2);

#ifdef __cplusplus
}
#endif /* _cplusplus */

#endif /* LBDBH_H */

/*@}*/

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

  /*  User-defined object ID is 4 ints long */
#define OBJ_ID_SZ 4

#if CMK_LBDB_ON
typedef struct {
  void *handle;            // pointer to LBDB
} LDHandle;
#else
typedef int LDHandle;
#endif

typedef struct _LDOMid {
  CkGroupID id;
#ifdef __cplusplus
  CmiBool operator==(const struct _LDOMid& omId) const {
    return (CmiBool)(id == omId.id);
  }
  inline void pup(PUP::er &p);
#endif
} LDOMid;

typedef struct {
  LDHandle ldb;
//  void *user_ptr;
  LDOMid id;
  int handle;		// index to LBOM
#ifdef __cplusplus
  inline void pup(PUP::er &p);
#endif
} LDOMHandle;

typedef struct _LDObjid {
  int id[OBJ_ID_SZ];
#ifdef __cplusplus
  CmiBool operator==(const struct _LDObjid& objid) const {
    for (int i=0; i<OBJ_ID_SZ; i++) if (id[i] != objid.id[i]) return CmiFalse;
    return CmiTrue;
  }
  inline void pup(PUP::er &p);
#endif
} LDObjid;

/* LDObjKey uniquely identify one object */
typedef struct _LDObjKey {
  LDOMid omId;
  LDObjid objId;
public:
#ifdef __cplusplus
  CmiBool operator==(const _LDObjKey& obj) const {
    return (CmiBool)(omId == obj.omId && objId == obj.objId);
  }
  inline LDOMid &omID() { return omId; }
  inline LDObjid &objID() { return objId; }
  inline const LDOMid &omID() const { return omId; }
  inline const LDObjid &objID() const { return objId; }
  inline void pup(PUP::er &p);
#endif
} LDObjKey;

typedef int LDObjIndex;
typedef int LDOMIndex;

typedef struct {
  LDOMHandle omhandle;
  LDObjid id;
  LDObjIndex  handle;
#ifdef __cplusplus
  inline const LDOMid &omID() const { return omhandle.id; }
  inline const LDObjid &objID() const { return id; }
  inline void pup(PUP::er &p);
#endif
} LDObjHandle;

typedef struct {
  LDObjHandle handle;
  double cpuTime;
  double wallTime;
  CmiBool migratable;
#ifdef __cplusplus
  inline const LDOMHandle &omHandle() const { return handle.omhandle; }
  inline const LDOMid &omID() const { return handle.omhandle.id; }
  inline const LDObjid &objID() const { return handle.id; }
  inline const LDObjid &id() const { return handle.id; }
  inline void pup(PUP::er &p);
#endif
} LDObjData;

/* used by load balancer */
typedef struct {
  int index;
  LDObjData data;
  int from_proc;
  int to_proc;
#ifdef __cplusplus
  inline void pup(PUP::er &p);
#endif
} LDObjStats;

#define LD_PROC_MSG      1
#define LD_OBJ_MSG       2
#define LD_OBJLIST_MSG   3

typedef struct _LDCommDesc {
  char type;
  union {
    int destProc;		/* 1:   processor level message */
    LDObjKey  destObj;		/* 2:   object based message    */
    struct {
      LDObjKey  *objs;
      int len;
    } destObjs;			/* 3:   one to many message     */
  } dest;
#ifdef __cplusplus
  char &get_type() { return type; }
  char get_type() const { return type; }
  int proc() const { return type==1?dest.destProc:-1; }
  LDObjKey &get_destObj() 
	{ CmiAssert(type==LD_OBJ_MSG); return dest.destObj; }
  LDObjKey const &get_destObj() const 
	{ CmiAssert(type==LD_OBJ_MSG); return dest.destObj; }
  LDObjKey * get_destObjs(int &len) 
	{ CmiAssert(type==LD_OBJLIST_MSG); len=dest.destObjs.len; return dest.destObjs.objs; }
  void init_objmsg(LDOMid &omid, LDObjid &objid) { 
	type=LD_OBJ_MSG; 
  	dest.destObj.omID()=omid;
  	dest.destObj.objID()=objid;
  }
  inline CmiBool operator==(const _LDCommDesc &obj) const {
    if (type != obj.type) return CmiFalse;
    switch (type) {
    case LD_PROC_MSG: return (CmiBool)(dest.destProc == obj.dest.destProc);
    case LD_OBJ_MSG:  return (CmiBool)(dest.destObj == obj.dest.destObj);
    case LD_OBJLIST_MSG: return CmiFalse;             // fixme
    }
    return CmiFalse;
  }
  inline void pup(PUP::er &p);
#endif
} LDCommDesc;

typedef struct {
  int src_proc;
  LDObjKey  sender;
  LDCommDesc   receiver;
  int messages;
  int bytes;
#ifdef __cplusplus
  inline int from_proc() const { return (src_proc != -1); }
  inline int recv_type() const { return receiver.get_type(); }
  inline void pup(PUP::er &p);
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

LDOMHandle LDRegisterOM(LDHandle _lbdb, LDOMid userID, 
			void *userptr, LDCallbacks cb);
void * LDOMUserData(LDOMHandle &_h);
void LDRegisteringObjects(LDOMHandle _h);
void LDDoneRegisteringObjects(LDOMHandle _h);

LDObjHandle LDRegisterObj(LDOMHandle h, LDObjid id, void *userptr,
			  int migratable);
void LDUnregisterObj(LDObjHandle h);

void * LDObjUserData(LDObjHandle &_h);
void LDObjTime(LDObjHandle &h, double walltime, double cputime);
int LDRunningObject(LDHandle _h, LDObjHandle* _o );
void LDObjectStart(const LDObjHandle &_h);
void LDObjectStop(const LDObjHandle &_h);
void LDSend(const LDOMHandle &destOM, const LDObjid &destid, unsigned int bytes);

void LDMessage(LDObjHandle from, 
	       LDOMid toOM, LDObjid *toID, int bytes);

void LDEstObjLoad(LDObjHandle h, double load);
void LDNonMigratable(const LDObjHandle &h);
void LDMigratable(const LDObjHandle &h);
void LDDumpDatabase(LDHandle _lbdb);

/*
 * Calls from load balancer to load database
 */  
typedef void (*LDMigratedFn)(void* data, LDObjHandle handle);
void LDNotifyMigrated(LDHandle _lbdb, LDMigratedFn fn, void* data);

typedef void (*LDStartLBFn)(void *user_ptr);
void LDAddStartLBFn(LDHandle _lbdb, LDStartLBFn fn, void* data);
void LDRemoveStartLBFn(LDHandle _lbdb, LDStartLBFn fn);
void LDStartLB(LDHandle _db);
void LDTurnManualLBOn(LDHandle _lbdb);
void LDTurnManualLBOff(LDHandle _lbdb);

void LDCollectStatsOn(LDHandle _lbdb);
void LDCollectStatsOff(LDHandle _lbdb);
void LDQueryEstLoad(LDHandle bdb);
void LDQueryKnownObjLoad(LDObjHandle &h, double *cpuT, double *wallT);

int LDGetObjDataSz(LDHandle _lbdb);
void LDGetObjData(LDHandle _lbdb, LDObjData *data);

int LDGetCommDataSz(LDHandle _lbdb);
void LDGetCommData(LDHandle _lbdb, LDCommData *data);

void LDBackgroundLoad(LDHandle _lbdb, double *walltime, double *cputime);
void LDIdleTime(LDHandle _lbdb, double *walltime);
void LDTotalTime(LDHandle _lbdb, double *walltime, double *cputime);

void LDClearLoads(LDHandle _lbdb);
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

LDBarrierClient LDAddLocalBarrierClient(LDHandle _lbdb,LDResumeFn fn,
					void* data);
void LDRemoveLocalBarrierClient(LDHandle _lbdb, LDBarrierClient h);
LDBarrierReceiver LDAddLocalBarrierReceiver(LDHandle _lbdb,LDBarrierFn fn,
					    void* data);
void LDRemoveLocalBarrierReceiver(LDHandle _lbdb,LDBarrierReceiver h);
void LDAtLocalBarrier(LDHandle _lbdb, LDBarrierClient h);
void LDLocalBarrierOn(LDHandle _db);
void LDLocalBarrierOff(LDHandle _db);
void LDResumeClients(LDHandle _lbdb);
int LDProcessorSpeed();
CmiBool LDOMidEqual(const LDOMid &i1, const LDOMid &i2);
CmiBool LDObjIDEqual(const LDObjid &i1, const LDObjid &i2);

/*
 *  LBDB Configuration calls
 */
void LDSetLBPeriod(LDHandle _db, double s);
double LDGetLBPeriod(LDHandle _db);

int LDMemusage(LDHandle _db);

#ifdef __cplusplus
}
#endif /* _cplusplus */

#ifdef __cplusplus
/* put outside of __cplusplus */
inline void LDOMid::pup(PUP::er &p) {
  id.pup(p);
}
PUPmarshall(LDOMid);
inline void LDObjid::pup(PUP::er &p) {
  for (int i=0; i<OBJ_ID_SZ; i++) p|id[i];
}
PUPmarshall(LDObjid);
inline void LDObjKey::pup(PUP::er &p) {
  p|omId;
  p|objId;
}
PUPmarshall(LDObjKey);
inline void LDObjStats::pup(PUP::er &p) {
  p|index;
  p|data;
  p|from_proc;
  p|to_proc;
}
PUPmarshall(LDObjStats);
inline void LDOMHandle::pup(PUP::er &p) {
  // skip ldb since it is a pointer
  p|id;
  p|handle;
}
PUPmarshall(LDOMHandle);
inline void LDObjHandle::pup(PUP::er &p) {
  p|omhandle;
  p|id;
  p|handle;
}
PUPmarshall(LDObjHandle);
inline void LDObjData::pup(PUP::er &p) {
  p|handle;
  p|cpuTime;
  p|wallTime;
  p|migratable;
}
PUPmarshall(LDObjData);
inline void LDCommDesc::pup(PUP::er &p) {
  p|type;
  switch (type) {
  case 1:  p|dest.destProc; break;
  case 2:  p|dest.destObj; break;
  case 3:  break;		// fixme
  }
}
PUPmarshall(LDCommDesc)
inline void LDCommData::pup(PUP::er &p) {
    p|src_proc;
    p|sender;
    p|receiver;
    p|messages;
    p|bytes;
}
PUPmarshall(LDCommData)
#endif

#endif /* LBDBH_H */

/*@}*/

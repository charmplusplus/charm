/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef LBDBH_H
#define LBDBH_H

#include "converse.h"
#include "charm.h"
#include "middle.h"

class LBDatabase;//Forward declaration

//typedef float floatType;
// type defined by build option
#ifndef CMK_LBTIME_TYPE
#define CMK_LBTIME_TYPE double
#endif
typedef CMK_LBTIME_TYPE LBRealType;

#define COMPRESS_LDB	1

extern int _lb_version;

#ifdef __cplusplus
extern "C" {
#endif

typedef void* cvoid; /* To eliminate warnings, because a C void* is not
			the same as a C++ void* */

  /*  User-defined object ID is 4 ints long (as defined in converse.h) */
  /*  as OBJ_ID_SZ */

#if CMK_LBDB_ON
typedef struct {
  void *handle;            // pointer to LBDB
} LDHandle;
#else
typedef int LDHandle;
#endif

typedef struct _LDOMid {
  CkGroupID id;
  CmiBool operator==(const struct _LDOMid& omId) const {
    return id == omId.id?CmiTrue:CmiFalse;
  }
  CmiBool operator<(const struct _LDOMid& omId) const {
    return id < omId.id?CmiTrue:CmiFalse;
  }
  CmiBool operator!=(const struct _LDOMid& omId) const {
    return id == omId.id?CmiFalse:CmiTrue;
  }
  inline void pup(PUP::er &p);
} LDOMid;

typedef struct {
  LDHandle ldb;
//  void *user_ptr;
  LDOMid id;
  int handle;		// index to LBOM
  inline void pup(PUP::er &p);
} LDOMHandle;

typedef struct _LDObjid {
  int id[OBJ_ID_SZ];
  CmiBool operator==(const struct _LDObjid& objid) const {
    for (int i=0; i<OBJ_ID_SZ; i++) if (id[i] != objid.id[i]) return CmiFalse;
    return CmiTrue;
  }
  CmiBool operator<(const struct _LDObjid& objid) const {
    for (int i=0; i<OBJ_ID_SZ; i++) {
      if (id[i] < objid.id[i]) return CmiTrue;
      else if (id[i] > objid.id[i]) return CmiFalse;
    }
    return CmiFalse;
  }
  inline void pup(PUP::er &p);
} LDObjid;

/* LDObjKey uniquely identify one object */
typedef struct _LDObjKey {
  /// Id of the location manager for this object
  LDOMid omId;
  LDObjid objId;
public:
  CmiBool operator==(const _LDObjKey& obj) const {
    return (CmiBool)(omId == obj.omId && objId == obj.objId);
  }
  CmiBool operator<(const _LDObjKey& obj) const {
    if (omId < obj.omId) return CmiTrue;
    else if (omId == obj.omId) return objId < obj.objId;
    else return CmiFalse;
  }
  inline LDOMid &omID() { return omId; }
  inline LDObjid &objID() { return objId; }
  inline const LDOMid &omID() const { return omId; }
  inline const LDObjid &objID() const { return objId; }
  inline void pup(PUP::er &p);
} LDObjKey;

typedef int LDObjIndex;
typedef int LDOMIndex;

typedef struct {
  LDOMHandle omhandle;
  LDObjid id;
  LDObjIndex  handle;
  inline const LDOMid &omID() const { return omhandle.id; }
  inline const LDObjid &objID() const { return id; }
  inline void pup(PUP::er &p);
} LDObjHandle;

typedef struct {
  LDObjHandle handle;
  LBRealType wallTime;
#if CMK_LB_CPUTIMER
  LBRealType cpuTime;
#endif
#if ! COMPRESS_LDB
  LBRealType minWall, maxWall;
#endif
  CmiBool migratable;
  CmiBool asyncArrival;
  inline const LDOMHandle &omHandle() const { return handle.omhandle; }
  inline const LDOMid &omID() const { return handle.omhandle.id; }
  inline const LDObjid &objID() const { return handle.id; }
  inline const LDObjid &id() const { return handle.id; }
  inline void pup(PUP::er &p);
} LDObjData;

/* used by load balancer */
typedef struct {
  int index;
  LDObjData data;
  int from_proc;
  int to_proc;
  inline void pup(PUP::er &p);
} LDObjStats;

#define LD_PROC_MSG      1
#define LD_OBJ_MSG       2
#define LD_OBJLIST_MSG   3

typedef struct _LDCommDesc {
  char type;
  union {
    int destProc;		/* 1:   processor level message */
    struct{
      LDObjKey  destObj;		/* 2:   object based message    */
      int destObjProc;
    } destObj;
    struct {
      LDObjKey  *objs;
      int len;
    } destObjs;			/* 3:   one to many message     */
  } dest;
  char &get_type() { return type; }
  char get_type() const { return type; }
  int proc() const { return type==LD_PROC_MSG?dest.destProc:-1; }
  void setProc(int pe) { CmiAssert(type==LD_PROC_MSG); dest.destProc = pe; }
  int lastKnown() const { 
    if (type==LD_OBJ_MSG) return dest.destObj.destObjProc;
    if (type==LD_PROC_MSG) return dest.destProc;
    return -1;
  }
  LDObjKey &get_destObj() 
	{ CmiAssert(type==LD_OBJ_MSG); return dest.destObj.destObj; }
  LDObjKey const &get_destObj() const 
	{ CmiAssert(type==LD_OBJ_MSG); return dest.destObj.destObj; }
  LDObjKey * get_destObjs(int &len) 
	{ CmiAssert(type==LD_OBJLIST_MSG); len=dest.destObjs.len; return dest.destObjs.objs; }
  void init_objmsg(LDOMid &omid, LDObjid &objid, int destObjProc) { 
	type=LD_OBJ_MSG; 
  	dest.destObj.destObj.omID()=omid;
  	dest.destObj.destObj.objID() =objid;
  	dest.destObj.destObjProc = destObjProc;
  }
  void init_mcastmsg(LDOMid &omid, LDObjid *objid, int len) { 
	type=LD_OBJLIST_MSG; 
	dest.destObjs.len = len;
	dest.destObjs.objs = new LDObjKey[len];
        for (int i=0; i<len; i++) {
  	  dest.destObjs.objs[i].omID()=omid;
  	  dest.destObjs.objs[i].objID() =objid[i];
	}
  }
  inline CmiBool operator==(const _LDCommDesc &obj) const;
  inline _LDCommDesc &operator=(const _LDCommDesc &c);
  inline void pup(PUP::er &p);
} LDCommDesc;

typedef struct {
  int src_proc;			// sender can either be a proc or an obj
  LDObjKey  sender;		// 
  LDCommDesc   receiver;
  int  sendHash, recvHash;
  int messages;
  int bytes;
  inline int from_proc() const { return (src_proc != -1); }
  inline int recv_type() const { return receiver.get_type(); }
  inline void pup(PUP::er &p);
  inline void clearHash() { sendHash = recvHash = -1; }
} LDCommData;

/*
 * Requests to load balancer
 *   FIXME: these routines don't seem to exist anywhere-- are they obsolete?
 *   Are the official versions now in LBDatabase.h?
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
typedef void (*LDAdaptResumeSyncFn) (LDObjHandle handle, int lb_ideal_period);

typedef struct {
  LDMigrateFn migrate;
  LDStatsFn setStats;
  LDQueryEstLoadFn queryEstLoad;
  LDAdaptResumeSyncFn adaptResumeSync;
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

void LDOMAdaptResumeSync(LDHandle _h, int lb_ideal_period);
void * LDOMUserData(LDOMHandle &_h);
void LDRegisteringObjects(LDOMHandle _h);
void LDDoneRegisteringObjects(LDOMHandle _h);

LDObjHandle LDRegisterObj(LDOMHandle h, LDObjid id, void *userptr,
			  int migratable);
void LDUnregisterObj(LDObjHandle h);
const LDObjHandle &LDGetObjHandle(LDHandle h, int idx);

void * LDObjUserData(LDObjHandle &_h);
void LDObjTime(LDObjHandle &h, LBRealType walltime, LBRealType cputime);
int  CLDRunningObject(LDHandle _h, LDObjHandle* _o );
void LDObjectStart(const LDObjHandle &_h);
void LDObjectStop(const LDObjHandle &_h);
void LDSend(const LDOMHandle &destOM, const LDObjid &destid, unsigned int bytes, int destObjProc, int force);
void LDMulticastSend(const LDOMHandle &destOM, LDObjid *destids, int ndests, unsigned int bytes, int nMsgs);

void LDMessage(LDObjHandle from, 
	       LDOMid toOM, LDObjid *toID, int bytes);

void LDEstObjLoad(LDObjHandle h, double load);
void LDNonMigratable(const LDObjHandle &h);
void LDMigratable(const LDObjHandle &h);
void LDAsyncMigrate(const LDObjHandle &h, CmiBool);
void LDDumpDatabase(LDHandle _lbdb);

/*
 * Calls from load balancer to load database
 */  
typedef void (*LDMigratedFn)(void* data, LDObjHandle handle, int waitBarrier);
void LDNotifyMigrated(LDHandle _lbdb, LDMigratedFn fn, void* data);

typedef void (*LDStartLBFn)(void *user_ptr);
void LDAddStartLBFn(LDHandle _lbdb, LDStartLBFn fn, void* data);
void LDRemoveStartLBFn(LDHandle _lbdb, LDStartLBFn fn);
void LDStartLB(LDHandle _db);
void LDTurnManualLBOn(LDHandle _lbdb);
void LDTurnManualLBOff(LDHandle _lbdb);

typedef void (*LDMigrationDoneFn)(void *user_ptr);
int LDAddMigrationDoneFn(LDHandle _lbdb, LDMigrationDoneFn fn,  void* data);
void  LDRemoveMigrationDoneFn(LDHandle _lbdb, LDMigrationDoneFn fn);
void LDMigrationDone(LDHandle _lbdb);

typedef void (*LDPredictFn)(void* user_ptr);
typedef void (*LDPredictModelFn)(void* user_ptr, void* model);
typedef void (*LDPredictWindowFn)(void* user_ptr, void* model, int wind);
void LDTurnPredictorOn(LDHandle _lbdb, void *model);
void LDTurnPredictorOnWin(LDHandle _lbdb, void *model, int wind);
void LDTurnPredictorOff(LDHandle _lbdb);
void LDChangePredictor(LDHandle _lbdb, void *model);
void LDCollectStatsOn(LDHandle _lbdb);
void LDCollectStatsOff(LDHandle _lbdb);
int  CLDCollectingStats(LDHandle _lbdb);
void LDQueryEstLoad(LDHandle bdb);
void LDGetObjLoad(LDObjHandle &h, LBRealType *wallT, LBRealType *cpuT);
void LDQueryKnownObjLoad(LDObjHandle &h, LBRealType *wallT, LBRealType *cpuT);

int LDGetObjDataSz(LDHandle _lbdb);
void LDGetObjData(LDHandle _lbdb, LDObjData *data);

int LDGetCommDataSz(LDHandle _lbdb);
void LDGetCommData(LDHandle _lbdb, LDCommData *data);

void LDBackgroundLoad(LDHandle _lbdb, LBRealType *walltime, LBRealType *cputime);
void LDIdleTime(LDHandle _lbdb, LBRealType *walltime);
void LDTotalTime(LDHandle _lbdb, LBRealType *walltime, LBRealType *cputime);
void LDGetTime(LDHandle _db, LBRealType *total_walltime,LBRealType *total_cputime,
                   LBRealType *idletime, LBRealType *bg_walltime, LBRealType *bg_cputime);

void LDClearLoads(LDHandle _lbdb);
int  LDMigrate(LDObjHandle h, int dest);
void LDMigrated(LDObjHandle h, int waitBarrier);

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

#if CMK_LBDB_ON
PUPbytes(LDHandle)
#endif

inline void LDOMid::pup(PUP::er &p) {
  id.pup(p);
}
PUPmarshall(LDOMid)

inline void LDObjid::pup(PUP::er &p) {
  for (int i=0; i<OBJ_ID_SZ; i++) p|id[i];
}
PUPmarshall(LDObjid)

inline void LDObjKey::pup(PUP::er &p) {
  p|omId;
  p|objId;
}
PUPmarshall(LDObjKey)

inline void LDObjStats::pup(PUP::er &p) {
  p|index;
  p|data;
  p|from_proc;
  p|to_proc;
}
PUPmarshall(LDObjStats)
inline void LDOMHandle::pup(PUP::er &p) {
  // skip ldb since it is a pointer
  int ptrSize = sizeof(void *);
  p|ptrSize;
  // if pointer size is not expected, must be in simulation mode
  // ignore this field
  if (p.isUnpacking() && ptrSize != sizeof(void *)) {  
    char dummy;
    for (int i=0; i<ptrSize; i++) p|dummy;
  }
  else
    p|ldb;
  p|id;
  p|handle;
}
PUPmarshall(LDOMHandle)

inline void LDObjHandle::pup(PUP::er &p) {
  p|omhandle;
  p|id;
  p|handle;
}
PUPmarshall(LDObjHandle)

inline void LDObjData::pup(PUP::er &p) {
  p|handle;
  p|wallTime;
#if CMK_LB_CPUTIMER
  p|cpuTime;
#endif
#if ! COMPRESS_LDB
  p|minWall;
  p|maxWall;
#endif
  p|migratable;
  if (_lb_version > -1) p|asyncArrival;
}
PUPmarshall(LDObjData)

inline CmiBool LDCommDesc::operator==(const LDCommDesc &obj) const {
    if (type != obj.type) return CmiFalse;
    switch (type) {
    case LD_PROC_MSG: return (CmiBool)(dest.destProc == obj.dest.destProc);
    case LD_OBJ_MSG:  return (CmiBool)(dest.destObj.destObj == obj.dest.destObj.destObj);
    case LD_OBJLIST_MSG: { if (dest.destObjs.len != obj.dest.destObjs.len) 
                               return CmiFalse;
                           for (int i=0; i<dest.destObjs.len; i++)
                             if (!(dest.destObjs.objs[i] == obj.dest.destObjs.objs[i])) return CmiFalse;
                           return CmiTrue; }
    }
    return CmiFalse;
}
inline LDCommDesc & LDCommDesc::operator=(const LDCommDesc &c) {
    type = c.type;
    switch (type) {
    case LD_PROC_MSG: dest.destProc = c.dest.destProc; break;
    case LD_OBJ_MSG:  dest.destObj.destObj = c.dest.destObj.destObj; break;
    case LD_OBJLIST_MSG: { dest.destObjs.len = c.dest.destObjs.len;
                           dest.destObjs.objs = new LDObjKey[dest.destObjs.len];
                           for (int i=0; i<dest.destObjs.len; i++)
                             dest.destObjs.objs[i] = c.dest.destObjs.objs[i]; 
                           break; }
    }
    return *this;
}
inline void LDCommDesc::pup(PUP::er &p) {
  p|type;
  switch (type) {
  case LD_PROC_MSG:  p|dest.destProc; break;
  case LD_OBJ_MSG:   p|dest.destObj.destObj; 
                     if (_lb_version == -1 && p.isUnpacking()) 
		       dest.destObj.destObjProc = -1;
		     else
		       p|dest.destObj.destObjProc; 
		     break;
  case LD_OBJLIST_MSG:  {  p|dest.destObjs.len; 
                           if (p.isUnpacking()) 
                               dest.destObjs.objs = new LDObjKey[dest.destObjs.len];
                           for (int i=0; i<dest.destObjs.len; i++) p|dest.destObjs.objs[i];
                           break; }
  }   // end of switch
}
PUPmarshall(LDCommDesc)

inline void LDCommData::pup(PUP::er &p) {
    p|src_proc;
    p|sender;
    p|receiver;
    p|messages;
    p|bytes;
    if (p.isUnpacking()) {
      sendHash = recvHash = -1;
    }
}
PUPmarshall(LDCommData)

#endif /* LBDBH_H */

/*@}*/

/**
 * \addtogroup CkLdb
*/
/*@{*/

#ifndef LBDBH_H
#define LBDBH_H

#include "converse.h"
#include "charm.h"
#include "middle.h"

#ifndef __STDC_FORMAT_MACROS
# define __STDC_FORMAT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
# define __STDC_LIMIT_MACROS
#endif
#include <inttypes.h>
#include <list>
#include <vector>

#include "pup_stl.h"

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
  bool operator==(const struct _LDOMid& omId) const {
    return id == omId.id?true:false;
  }
  bool operator<(const struct _LDOMid& omId) const {
    return id < omId.id?true:false;
  }
  bool operator!=(const struct _LDOMid& omId) const {
    return id == omId.id?false:true;
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

/* LDObjKey uniquely identify one object */
typedef struct _LDObjKey {
  /// Id of the location manager for this object
  LDOMid omId;
  CmiUInt8 objId;
public:
  _LDObjKey() : omId(), objId() { }
  _LDObjKey(LDOMid &om, CmiUInt8 obj) : omId(om), objId(obj) { }
  bool operator==(const _LDObjKey& obj) const {
    return (bool)(omId == obj.omId && objId == obj.objId);
  }
  bool operator<(const _LDObjKey& obj) const {
    if (omId < obj.omId) return true;
    else if (omId == obj.omId) return objId < obj.objId;
    else return false;
  }
  inline LDOMid &omID() { return omId; }
  inline CmiUInt8 &objID() { return objId; }
  inline const LDOMid &omID() const { return omId; }
  inline const CmiUInt8 &objID() const { return objId; }
  inline void pup(PUP::er &p);
} LDObjKey;

typedef int LDObjIndex;
typedef int LDOMIndex;

typedef struct {
  LDOMHandle omhandle;
  CmiUInt8 id;
  LDObjIndex  handle;
  inline const LDOMid &omID() const { return omhandle.id; }
  inline const CmiUInt8 &objID() const { return id; }
  inline void pup(PUP::er &p);
} LDObjHandle;

/* defines user data layout  */
class LBUserDataLayout {
int length;
int count;
public:
  LBUserDataLayout(): length(0), count(0) {}
  int claim(int size) {
    count++;
    int oldlen = length;
    length+=size;
    return oldlen;
  }
  int size() { return length; }
};

CkpvExtern(LBUserDataLayout, lbobjdatalayout);

class LBObjUserData {
  std::vector<char> data;
public:
  LBObjUserData() {
    data.resize(CkpvAccess(lbobjdatalayout).size());
  }
  LBObjUserData(const LBObjUserData &d) {
    this->data = d.data;
  }
  LBObjUserData(LBObjUserData &&d) {
    this->data = std::move(d.data);
  }

  ~LBObjUserData() { }
  LBObjUserData &operator = (const LBObjUserData &d) {
    this->data = d.data;
    return *this;
  }
  LBObjUserData &operator = (LBObjUserData &&d) {
    this->data = std::move(d.data);
    return *this;
  }
  inline void pup(PUP::er &p);
  void *getData(int idx) { return data.data()+idx; }
};

typedef struct {
  LDObjHandle handle;
  LBRealType wallTime;
#if CMK_LB_CPUTIMER
  LBRealType cpuTime;
#endif
#if ! COMPRESS_LDB
  LBRealType minWall, maxWall;
#endif
  bool migratable;
  bool asyncArrival;
#if CMK_LB_USER_DATA
  LBObjUserData   userData;
#endif
  // An encoded approximation of the amount of data the object would pack;
  // call pup_decodeSize(pupSize) to get the actual approximate value
  CmiUInt2 pupSize;
  inline const LDOMHandle &omHandle() const { return handle.omhandle; }
  inline const LDOMid &omID() const { return handle.omhandle.id; }
  inline const CmiUInt8 &objID() const { return handle.id; }
  inline const CmiUInt8 &id() const { return handle.id; }
  inline void pup(PUP::er &p);
#if CMK_LB_USER_DATA
  void* getUserData(int idx)  { return userData.getData(idx); }
#endif
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
  _LDCommDesc()
    : type(LD_PROC_MSG)
    , dest()
  { }
  _LDCommDesc(LDOMid &omid, CmiUInt8 objid, int destObjProc) // init_objmsg
    : type(LD_OBJ_MSG)
    , dest(omid, objid, destObjProc)
  { }
  _LDCommDesc(LDOMid &omid, CmiUInt8 *objid, int len) // init_mcastmsg
    : type(LD_OBJLIST_MSG)
    , dest(omid, objid, len)
  { }
  _LDCommDesc(const _LDCommDesc &rhs)
  {
    *this = rhs;
  }

  char type{};

  union destunion {
    destunion() : destProc() {}
    destunion(LDOMid &omid, CmiUInt8 objid, int destObjProc)
     : destObj(omid, objid, destObjProc)
    { }
    destunion(LDOMid &omid, CmiUInt8 *objid, int len)
     : destObjs()
    {
      destObjs.reserve(len);
      for (int i=0; i<len; i++)
        destObjs.emplace_back(omid, objid[i]);
	  }
    ~destunion() {}

    int destProc;		/* 1:   processor level message */

    struct objstruct {
      objstruct(LDOMid &omid, CmiUInt8 objid, int destObjProc_)
        : destObj(omid, objid)
        , destObjProc(destObjProc_)
      { }

      LDObjKey  destObj;		/* 2:   object based message    */
      int destObjProc;
    } destObj;

    std::vector<LDObjKey> destObjs; /* 3:   one to many message     */
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
  const LDObjKey * get_destObjs (int &len) const
	{ CmiAssert(type==LD_OBJLIST_MSG); len=dest.destObjs.size(); return dest.destObjs.data(); }

  inline bool operator==(const _LDCommDesc &obj) const;
  inline _LDCommDesc &operator=(const _LDCommDesc &c);
  inline void pup(PUP::er &p);
} LDCommDesc;

typedef struct _LDCommData {
  int src_proc;			// sender can either be a proc or an obj
  LDObjKey  sender;		// 
  LDCommDesc   receiver;
  int  sendHash, recvHash;
  int messages;
  int bytes;
  inline _LDCommData &operator=(const _LDCommData &o) {
    if (&o == this) return *this;
    src_proc = o.src_proc;
    sender = o.sender; receiver = o.receiver;
    sendHash = o.sendHash; recvHash = o.recvHash;
    messages = o.messages;
    bytes = o.bytes;
    return *this;
  }
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
typedef void (*LDMetaLBResumeWaitingCharesFn) (LDObjHandle handle, int lb_ideal_period);
typedef void (*LDMetaLBCallLBOnCharesFn) (LDObjHandle handle);

typedef struct {
  LDMigrateFn migrate;
  LDStatsFn setStats;
  LDQueryEstLoadFn queryEstLoad;
  LDMetaLBResumeWaitingCharesFn metaLBResumeWaitingChares;
  LDMetaLBCallLBOnCharesFn metaLBCallLBOnChares;
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
void LDUnregisterOM(LDHandle _db, LDOMHandle handle);

void LDOMMetaLBResumeWaitingChares(LDHandle _h, int lb_ideal_period);
void LDOMMetaLBCallLBOnChares(LDHandle _h);
void * LDOMUserData(LDOMHandle &_h);
void LDRegisteringObjects(LDOMHandle _h);
void LDDoneRegisteringObjects(LDOMHandle _h);

LDObjHandle LDRegisterObj(LDOMHandle h, CmiUInt8 id, void *userptr,
			  bool migratable);
void LDUnregisterObj(LDObjHandle h);

void *LDObjUserData(LDObjHandle &_h);
#if CMK_LB_USER_DATA
void *LDDBObjUserData(LDObjHandle &_h, int idx);
#endif
void LDObjTime(LDObjHandle &h, LBRealType walltime, LBRealType cputime);
int  CLDRunningObject(LDHandle _h, LDObjHandle* _o );
void LDObjectStart(const LDObjHandle &_h);
void LDObjectStop(const LDObjHandle &_h);
void LDSend(const LDOMHandle &destOM, const CmiUInt8 &destid, unsigned int bytes, int destObjProc, int force);
void LDMulticastSend(const LDOMHandle &destOM, CmiUInt8 *destids, int ndests, unsigned int bytes, int nMsgs);

void LDMessage(LDObjHandle from, 
	       LDOMid toOM, CmiUInt8 *toID, int bytes);

void LDEstObjLoad(LDObjHandle h, double load);
void LDNonMigratable(const LDObjHandle &h);
void LDMigratable(const LDObjHandle &h);
void LDSetPupSize(const LDObjHandle &h, size_t);
void LDAsyncMigrate(const LDObjHandle &h, bool);
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
void LDGetCommInfo(LDHandle _lbdb, int& bytes, int& msgs, int& withinbytes, int& outsidebytes, int& num_nghbors, int& hops, int& hopbytes);

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

class client;
struct LDBarrierClient {
  std::list<client *>::iterator i;
  LDBarrierClient() { }
  LDBarrierClient(std::list<client *>::iterator in)
  : i(in) { }
};

class receiver;
struct LDBarrierReceiver {
  std::list<receiver *>::iterator i;
  LDBarrierReceiver() { }
  LDBarrierReceiver(std::list<receiver *>::iterator in)
  : i(in) { }
};

void LDAtLocalBarrier(LDHandle _lbdb, LDBarrierClient h);
void LDDecreaseLocalBarrier(LDHandle _lbdb, LDBarrierClient h, int c);
void LDLocalBarrierOn(LDHandle _db);
void LDLocalBarrierOff(LDHandle _db);
void LDResumeClients(LDHandle _lbdb);
int LDProcessorSpeed();
bool LDOMidEqual(const LDOMid &i1, const LDOMid &i2);

/*
 *  LBDB Configuration calls
 */
void LDSetLBPeriod(LDHandle _db, double s);
double LDGetLBPeriod(LDHandle _db);

int LDMemusage(LDHandle _db);

#ifdef __cplusplus
}
#endif /* _cplusplus */

const LDObjHandle &LDGetObjHandle(LDHandle h, int idx);
LDBarrierClient LDAddLocalBarrierClient(LDHandle _lbdb,LDResumeFn fn,
					void* data);
void LDRemoveLocalBarrierClient(LDHandle _lbdb, LDBarrierClient h);
LDBarrierReceiver LDAddLocalBarrierReceiver(LDHandle _lbdb,LDBarrierFn fn,
					    void* data);
void LDRemoveLocalBarrierReceiver(LDHandle _lbdb,LDBarrierReceiver h);

#if CMK_LBDB_ON
PUPbytes(LDHandle)
#endif

inline void LDOMid::pup(PUP::er &p) {
  id.pup(p);
}
PUPmarshall(LDOMid)

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

inline void LBObjUserData::pup(PUP::er &p) {
  p|data;
}
PUPmarshall(LBObjUserData)

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
#if CMK_LB_USER_DATA
  if (_lb_version > 2) {
    p|userData;
  }
#endif
  p|pupSize;
}
PUPmarshall(LDObjData)

inline bool LDCommDesc::operator==(const LDCommDesc &obj) const {
    if (type != obj.type) return false;
    switch (type) {
    case LD_PROC_MSG: return dest.destProc == obj.dest.destProc;
    case LD_OBJ_MSG:  return dest.destObj.destObj == obj.dest.destObj.destObj;
    case LD_OBJLIST_MSG: return dest.destObjs == obj.dest.destObjs;
    }
    return false;
}
inline LDCommDesc & LDCommDesc::operator=(const LDCommDesc &c) {
    type = c.type;
    switch (type) {
    case LD_PROC_MSG: dest.destProc = c.dest.destProc; break;
    case LD_OBJ_MSG:  dest.destObj = c.dest.destObj; break;
    case LD_OBJLIST_MSG: dest.destObjs = c.dest.destObjs; break;
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
  case LD_OBJLIST_MSG: p|dest.destObjs; break;
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

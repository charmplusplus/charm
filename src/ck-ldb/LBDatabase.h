#ifndef LBDATABASE_H
#define LBDATABASE_H

#include "lbdb.h"

#include "LBObj.h"
#include "LBOM.h"
#include "LBComm.h"
#include "LBMachineUtil.h"

#if CMK_CUDA && CMK_LBDB_ON
// For hapiCuptiStartTracing/hapiCuptiStopTracing, called from TurnStatsOn/Off.
// Declared here rather than by including hapi.h, which drags the CUDA runtime
// headers into every consumer of LBDatabase.h.
void hapiCuptiStartTracing();
void hapiCuptiStopTracing();
#endif

#include <unordered_map>
#include <vector>

class CkSyncBarrier;

class LBDatabase {
friend class LBManager;
private:
  LBDatabase();
  struct LBObjEntry {
    static const LDObjIndex DEFAULT_NEXT = -1;
    LBObj* obj;
    LDObjIndex nextEmpty;

    LBObjEntry(LBObj* obj, LDObjIndex nextEmpty = DEFAULT_NEXT) : obj(obj), nextEmpty(nextEmpty) {}
  };

  std::vector<LBObjEntry> objs;
  std::vector<LBOM*> oms;
  LDObjIndex objsEmptyHead;
  int omCount;
  int omsRegistering;
  LBCommTable* commTable;
  bool statsAreOn;
  double obj_walltime;
  LBMachineUtil machineUtil;
  CkSyncBarrier* syncBarrier;

#if CMK_LB_CPUTIMER
  double obj_cputime;
#endif

public:
  inline void MeasuredObjTime(double wtime, double ctime) {
    if (statsAreOn) {
      obj_walltime += wtime;
#if CMK_LB_CPUTIMER
      obj_cputime += ctime;
#endif
    }
  }
  inline LBOM* LbOM(LDOMHandle h) {
    return oms[h.handle];
  }
  inline LBObj *LbObj(const LDObjHandle &h) const {
    return objs[h.handle].obj;
  }
  inline LBObj *LbObjIdx(int h) const {
    return objs[h].obj;
  }

  inline void ObjTime(LDObjHandle h, double walltime, double cputime) {
    LbObj(h)->IncrementTime(walltime, cputime);
    MeasuredObjTime(walltime, cputime);
  };

  inline void GetObjLoad(LDObjHandle &h, LBRealType &walltime, LBRealType &cputime) {
    LbObj(h)->getTime(&walltime, &cputime);
  };

  inline void GetObjGPULoad(LDObjHandle &h, LBRealType &gputime) {
    LbObj(h)->getGPUTime(&gputime);
  };

  // Copy the per-object normalized GPU loads computed by
  // hapiNormalizeCuptiLoads into this PE's LB objects. The map is shared
  // per-process and every PE reads it concurrently, so this must not mutate it.
  // An object whose application declared its load this interval keeps the
  // declaration (LBObj::gpuDeclared).
  inline void SetObjGPULoad(
      const std::unordered_map<LDObjKey, double, LDObjKeyHash> &id_loadMap)
  {
    for (size_t i = 0; i < objs.size(); i++) {
      if (objs[i].obj == nullptr)
        continue;
      if (objs[i].obj->hasGPUDeclared())
        continue;
      const LDObjHandle &handle = objs[i].obj->GetLDObjHandle();
      LDObjKey key;
      key.omID() = handle.omID();
      key.objID() = handle.objID();
      auto it = id_loadMap.find(key);
      if (it == id_loadMap.end())
        continue;
      objs[i].obj->setGPUTiming(it->second);
    }
  }

  inline void ObjPosition(LDObjHandle h, const std::vector<LBRealType>& pos) {
    LbObj(h)->setPosition(pos);
  };
  inline const std::vector<LBRealType>& GetObjPosition(LDObjHandle h) {
    return LbObj(h)->getPosition();
  };

  // Replication declaration; see LDObjData::replicaGroup.
  inline void ObjReplica(LDObjHandle h, int group, int rank, int count, int maxCount, size_t extraBytes,
                         double costNode = 0.0, double costRemote = 0.0, int parts = 0,
                         size_t exchangeBytes = 0, double exchangeMsgs = 0.0, size_t extraPerPeer = 0) {
    LbObj(h)->setReplica(group, rank, count, maxCount, extraBytes, costNode, costRemote, parts,
                         exchangeBytes, exchangeMsgs, extraPerPeer);
  };
  inline void GetObjReplica(LDObjHandle h, int& group, int& rank, int& count, int& maxCount, size_t& extraBytes) {
    LbObj(h)->getReplica(group, rank, count, maxCount, extraBytes);
  };

  inline void* GetObjUserData(LDObjHandle &h) {
    return LbObj(h)->getLocalUserData();
  }
  // GPU activity tracing follows the same switch as CPU instrumentation, so an
  // application that calls LBTurnInstrumentOn()/LBTurnInstrumentOff() around
  // its own AtSync schedule controls both with one call. Tracing is by far the
  // more expensive of the two, so leaving it off between load-balancing steps
  // is what makes instrumented runs affordable.
  inline void TurnStatsOn(void)
       {statsAreOn = true; machineUtil.StatsOn();
#if CMK_CUDA && CMK_LBDB_ON
        hapiCuptiStartTracing();
#endif
       }
  inline void TurnStatsOff(void)
       {statsAreOn = false; machineUtil.StatsOff();
#if CMK_CUDA && CMK_LBDB_ON
        hapiCuptiStopTracing();
#endif
       }
  inline bool StatsOn(void) const
       { return statsAreOn; };
  inline void IdleTime(LBRealType *walltime) {
    machineUtil.IdleTime(walltime);
  }
  inline void TotalTime(LBRealType *walltime, LBRealType *cputime) {
    machineUtil.TotalTime(walltime, cputime);
  }

  inline void QueryKnownObjLoad(LDObjHandle &h, LBRealType &walltime, LBRealType &cputime) {
    LbObj(h)->lastKnownLoad(&walltime, &cputime);
  };
  inline void NonMigratable(LDObjHandle h) { LbObj(h)->SetMigratable(false); };
  inline void Migratable(LDObjHandle h) { LbObj(h)->SetMigratable(true); };
  inline void setPupSize(LDObjHandle h, size_t pup_size) { LbObj(h)->setPupSize(pup_size);};
  inline void setGPUPupSize(LDObjHandle h, size_t gpu_pup_size) { LbObj(h)->setGPUPupSize(gpu_pup_size);};
  inline void UseAsyncMigrate(LDObjHandle h, bool flag) { LbObj(h)->UseAsyncMigrate(flag); };
  inline int GetCommDataSz(void) {
    if (commTable)
      return commTable->CommCount();
    else return 0;
  }

  inline void GetCommData(LDCommData *data) {
    if (commTable) commTable->GetCommData(data);
  }

  inline void GetCommInfo(int& bytes, int& msgs, int& withinbytes, int& outsidebytes, int& num_nghbors, int& hops, int& hopbytes) {
    if (commTable)
      commTable->GetCommInfo(bytes, msgs, withinbytes, outsidebytes, num_nghbors, hops, hopbytes);
  }


  LDOMHandle RegisterOM(LDOMid userID, void *userptr, LDCallbacks cb);
  int Migrate(LDObjHandle h, int dest);
  // The balancer's replication decisions, delivered to the object's manager
  // (LDCallbacks::clone / retire). Return 1 when the object was found, 0 when
  // the handle no longer names it (it died, or its slot was reused since the
  // statistics the decision was planned from).
  int Clone(LDObjHandle h, int dest, int rank);
  int Retire(LDObjHandle h);
  int CloneNotice(LDObjHandle h, int dest, int rank);
  int RetireNotice(LDObjHandle h, int rank);
private:
  LBObj* replicaTarget(const LDObjHandle& h, const char* what) const;
public:
  void UnregisterOM(LDOMHandle omh);
  void RegisteringObjects(LDOMHandle omh);
  void DoneRegisteringObjects(LDOMHandle omh);
  int GetObjDataSz(void);
  void GetObjData(LDObjData *data);
  void MetaLBCallLBOnChares();
  void MetaLBResumeWaitingChares(int lb_period);
  void ClearLoads(void);
  int useMem(void);
  LDObjHandle RegisterObj(LDOMHandle omh, CmiUInt8 id, void* userPtr,
                          int migratable);
  void UnregisterObj(LDObjHandle h);
  void EstObjLoad(const LDObjHandle &h, double cpuload);
  void EstObjGPULoad(const LDObjHandle &h, double gpuload);
  void BackgroundLoad(LBRealType *walltime, LBRealType *cputime);
  void Send(const LDOMHandle &destOM, const CmiUInt8 &destID, unsigned int bytes, int destObjProc, int force = 0);
  void MulticastSend(const LDOMHandle &_om, CmiUInt8 *_ids, int _n, unsigned int _b, int _nMsgs=1);
  void GetTime(LBRealType *total_walltime, LBRealType *total_cputime,
               LBRealType *idletime, LBRealType *bg_walltime,
               LBRealType *bg_cputime);
  const std::vector<LBObjEntry>& getObjs() {return objs;}

  inline void ObjectStart(const LDObjHandle &h) {
    if (StatsOn()) {
      LbObj(h)->StartTimer();
    }
  };

  inline void ObjectStop(const LDObjHandle &h) {
    LBObj* const obj = LbObj(h);

    if (StatsOn()) {
      LBRealType walltime, cputime;
      obj->StopTimer(&walltime, &cputime);
      obj->IncrementTime(walltime, cputime);
      MeasuredObjTime(walltime, cputime);
    }
  };
  inline const LDObjHandle &GetObjHandle(int idx) {
    return LbObjIdx(idx)->GetLDObjHandle();
  }
  // The handle of the object in slot idx, when the slot still holds the
  // object with this id (GetObjHandle dereferences an empty slot).
  inline bool LiveObjHandle(int idx, CmiUInt8 id, LDObjHandle& h) const {
    if (idx < 0 || (size_t)idx >= objs.size() || objs[idx].obj == nullptr) return false;
    h = objs[idx].obj->GetLDObjHandle();
    return h.id == id;
  }
  void CollectStatsOn(void);
  inline void CollectStatsOff(void) { TurnStatsOff(); };
  inline int  CollectingStats(void) {
  #if CMK_LBDB_ON
    return StatsOn();
  #else
    return 0;
  #endif
  };
};

#endif /* LBDATABASE_H */

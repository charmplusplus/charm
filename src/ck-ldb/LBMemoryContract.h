#ifndef LB_MEMORY_CONTRACT_H
#define LB_MEMORY_CONTRACT_H

// The LB memory contract: the strategy-side interface that discharges the
// migration runtime's two per-device preconditions,
//
//   I-final:  u0(g) - sum_{src=g} f(m) + sum_{dst=g} f(m)  <=  (1-eps) * C_g
//   I-batch:  sum_{src=g, batch} sigma(m)                  <=  S_g
//
// for every device g, where f is a chare's resident device footprint and
// sigma its serialized (staged) size. Any strategy whose emitted move set
// satisfies these inherits the runtime's no-OOM / no-deadlock guarantee; the
// ContractVerifier below repairs the output of strategies that do not use
// the ledger themselves.
//
// Header-only: strategies and the central LB include this directly.

#include "BaseLB.h"
#include "CentralLB.h"
#include <vector>
#include <unordered_map>
#include <cstdlib>

CkpvExtern(int, _lb_obj_index);  // the footprint slot (see _loadbalancerInit)

// ---------------------------------------------------------------------------
// LBMemoryModel: one uniform view over what the contract needs, built from
// the stats every central strategy already receives. PEs are grouped into
// devices by gpu_device_id, mirroring the GPU-group construction the GPU
// strategies use.
// ---------------------------------------------------------------------------
class LBMemoryModel {
public:
  struct Device {
    uint64_t gpu_id;
    size_t memFree;       // free device memory at stats time (cudaMemGetInfo)
    size_t stagingFree;   // staging (LB pool) free bytes at stats time
    std::vector<int> pes; // PEs mapped to this device
  };

  void build(BaseLB::LDStats* stats) {
    devices_.clear();
    devToIdx_.clear();
    peToDev_.assign(stats->nprocs(), -1);
    for (int pe = 0; pe < stats->nprocs(); pe++) {
      if (!stats->procs[pe].available) continue;
      uint64_t id = stats->procs[pe].gpu_device_id;
      auto it = devToIdx_.find(id);
      int d;
      if (it == devToIdx_.end()) {
        d = (int)devices_.size();
        devToIdx_[id] = d;
        Device dev;
        dev.gpu_id = id;
        dev.memFree = stats->procs[pe].gpu_mem_remaining;
        dev.stagingFree = stats->procs[pe].pool_buff_mem_remaining;
        devices_.push_back(dev);
      } else {
        d = it->second;
      }
      devices_[d].pes.push_back(pe);
      peToDev_[pe] = d;
    }
    stats_ = stats;
  }

  int numDevices() const { return (int)devices_.size(); }
  const Device& device(int d) const { return devices_[d]; }
  int deviceOfPe(int pe) const {
    return (pe >= 0 && pe < (int)peToDev_.size()) ? peToDev_[pe] : -1;
  }
  int deviceIndexOf(uint64_t gpu_id) const {
    auto it = devToIdx_.find(gpu_id);
    return it == devToIdx_.end() ? -1 : it->second;
  }

  // Serialized (staged) size of object i: what one migration stages.
  size_t stagedSize(int i) const { return (size_t)stats_->objData[i].gpuPupSize; }

  // Resident device footprint of object i: the user-data slot the AtSync
  // producer fills, floored at the serialized size so a missing producer can
  // never read as "free to move".
  size_t footprint(int i) const {
    size_t f = 0;
#if CMK_LB_USER_DATA
    if (CkpvAccess(_lb_obj_index) >= 0)
      f = *(size_t*)stats_->objData[i].getUserData(CkpvAccess(_lb_obj_index));
#endif
    size_t s = stagedSize(i);
    return f > s ? f : s;
  }

  // Two migrating endpoints on one physical process use a held transport
  // (peer copies with the source retained until completion); everything else
  // is staged. Held moves get no source credit in the ledger.
  static bool heldTransport(int fromPe, int toPe) {
    return CmiNodeOf(fromPe) == CmiNodeOf(toPe);
  }

private:
  std::vector<Device> devices_;
  std::unordered_map<uint64_t, int> devToIdx_;
  std::vector<int> peToDev_;
  BaseLB::LDStats* stats_ = nullptr;
};

// ---------------------------------------------------------------------------
// MemoryLedger: transactional feasibility for planning. Encodes the
// accounting a strategy must not get wrong: staged moves credit the source at
// pack time; held transports credit nothing until completion; staging is
// debited on both ends while the runtime lands payloads in pool blocks.
// ---------------------------------------------------------------------------
class MemoryLedger {
public:
  // headroom: the (1 - eps) margin applied to free device memory.
  // waveStaging: while migration waves execute unbatched, the whole wave's
  // staging demand must fit the reserve, so sigma is debited per move; once
  // batch execution is active this becomes the batch planner's job and the
  // per-move debit is disabled.
  void init(const LBMemoryModel* model, double headroom = 0.95,
            bool waveStaging = true) {
    model_ = model;
    waveStaging_ = waveStaging;
    memAvail_.resize(model->numDevices());
    stagingAvail_.resize(model->numDevices());
    for (int d = 0; d < model->numDevices(); d++) {
      // Environment override for adversarial testing: cap every device's
      // planning headroom at CHARM_LB_MEM_CAP_MB regardless of what the
      // device reports.
      size_t mem = (size_t)((double)model->device(d).memFree * headroom);
      const char* cap = getenv("CHARM_LB_MEM_CAP_MB");
      if (cap != NULL) {
        size_t capB = (size_t)atol(cap) * 1024 * 1024;
        if (mem > capB) mem = capB;
      }
      memAvail_[d] = mem;
      stagingAvail_[d] = model->device(d).stagingFree;
    }
  }

  bool feasible(int obj, int fromPe, int toPe) const {
    int s = model_->deviceOfPe(fromPe), d = model_->deviceOfPe(toPe);
    if (s < 0 || d < 0 || s == d) return true;  // no device change: no memory moves
    size_t f = model_->footprint(obj);
    size_t sig = model_->stagedSize(obj);
    if (sig == 0 && f == 0) return true;        // no device state
    bool held = LBMemoryModel::heldTransport(fromPe, toPe);
    if (memAvail_[d] < f) return false;
    if (!held && waveStaging_ &&
        (stagingAvail_[s] < sig || stagingAvail_[d] < sig))
      return false;
    if (!held && sig > model_->device(s).stagingFree)
      return false;                             // could never be packed at all
    return true;
  }

  void commit(int obj, int fromPe, int toPe) {
    int s = model_->deviceOfPe(fromPe), d = model_->deviceOfPe(toPe);
    if (s < 0 || d < 0 || s == d) return;
    size_t f = model_->footprint(obj);
    size_t sig = model_->stagedSize(obj);
    bool held = LBMemoryModel::heldTransport(fromPe, toPe);
    memAvail_[d] -= (memAvail_[d] >= f) ? f : memAvail_[d];
    if (!held) {
      memAvail_[s] += f;  // departure frees at pack: the two-phase credit
      if (waveStaging_) {
        stagingAvail_[s] -= (stagingAvail_[s] >= sig) ? sig : stagingAvail_[s];
        stagingAvail_[d] -= (stagingAvail_[d] >= sig) ? sig : stagingAvail_[d];
      }
    }
  }

  void rollback(int obj, int fromPe, int toPe) {
    int s = model_->deviceOfPe(fromPe), d = model_->deviceOfPe(toPe);
    if (s < 0 || d < 0 || s == d) return;
    size_t f = model_->footprint(obj);
    size_t sig = model_->stagedSize(obj);
    bool held = LBMemoryModel::heldTransport(fromPe, toPe);
    memAvail_[d] += f;
    if (!held) {
      memAvail_[s] -= (memAvail_[s] >= f) ? f : memAvail_[s];
      if (waveStaging_) {
        stagingAvail_[s] += sig;
        stagingAvail_[d] += sig;
      }
    }
  }

  size_t memAvailOn(int dev) const { return memAvail_[dev]; }
  size_t stagingAvailOn(int dev) const { return stagingAvail_[dev]; }

private:
  const LBMemoryModel* model_ = nullptr;
  bool waveStaging_ = true;
  std::vector<size_t> memAvail_;
  std::vector<size_t> stagingAvail_;
};

// ---------------------------------------------------------------------------
// ContractVerifier: hardens ANY strategy's finished move list. Recomputes the
// contract over the moves and repairs violations by refusing moves --
// largest-footprint first on the offending device, so the fewest moves are
// lost. A refused move keeps its chare where it is, which is always feasible.
// Returns the number of refused moves; refused entries have to_pe set back to
// from_pe.
// ---------------------------------------------------------------------------
class ContractVerifier {
public:
  // moves reference objects by stats index; from/to are PEs.
  struct Move {
    int obj;
    int fromPe;
    int* toPe;  // points into the strategy's decision storage, edited on refusal
  };

  static int verifyAndRepair(BaseLB::LDStats* stats, std::vector<Move>& moves,
                             double headroom = 0.95) {
    LBMemoryModel model;
    model.build(stats);
    if (model.numDevices() == 0) return 0;
    MemoryLedger ledger;
    ledger.init(&model, headroom);

    const bool dbg = (getenv("CHARM_DEBUG_MEMCONTRACT") != NULL);
    int refused = 0;

    // Commit cheapest-feasibility-order: smallest footprint first, so large
    // offenders are what remains when a device runs out and get refused.
    std::vector<int> order(moves.size());
    for (size_t i = 0; i < moves.size(); i++) order[i] = (int)i;
    for (size_t a = 0; a < order.size(); a++)      // insertion sort: lists are small
      for (size_t b = a + 1; b < order.size(); b++)
        if (model.footprint(moves[order[b]].obj) <
            model.footprint(moves[order[a]].obj)) {
          int t = order[a]; order[a] = order[b]; order[b] = t;
        }

    for (size_t k = 0; k < order.size(); k++) {
      Move& m = moves[order[k]];
      if (*m.toPe == m.fromPe) continue;
      if (ledger.feasible(m.obj, m.fromPe, *m.toPe)) {
        ledger.commit(m.obj, m.fromPe, *m.toPe);
      } else {
        if (dbg)
          CkPrintf("[%d] memcontract: refusing move of obj %d (%zu bytes) "
                   "pe %d -> %d\n",
                   CkMyPe(), m.obj, model.footprint(m.obj), m.fromPe, *m.toPe);
        *m.toPe = m.fromPe;
        refused++;
      }
    }
    if (dbg && refused)
      CkPrintf("[%d] memcontract: refused %d move(s) to preserve the "
               "memory contract\n", CkMyPe(), refused);
    return refused;
  }
};

// ---------------------------------------------------------------------------
// LBBatchPlanner: discharges I-batch. Partitions the decided move list so
// each batch's staged bytes fit every device's staging reserve (both ends
// debited, matching the unbatched runtime's pool use). Held transports stage
// nothing and always ride the first batch. Greedy first-fit; returns the
// number of batches, and batchOf[i] for every object whose to != from.
// CHARM_LB_FORCE_BATCHES=N round-robins moves into N batches regardless of
// staging -- the protocol-test knob.
// ---------------------------------------------------------------------------
class LBBatchPlanner {
public:
  static int plan(BaseLB::LDStats* stats, std::vector<int>& batchOf) {
    batchOf.assign(stats->objData.size(), 0);

    std::vector<int> moved;
    for (int i = 0; i < (int)stats->objData.size(); i++)
      if (stats->to_proc[i] != stats->from_proc[i]) moved.push_back(i);
    if (moved.empty()) return 1;

    const char* force = getenv("CHARM_LB_FORCE_BATCHES");
    if (force != NULL && atoi(force) > 1) {
      int n = atoi(force);
      for (size_t k = 0; k < moved.size(); k++) batchOf[moved[k]] = (int)k % n;
      return n;
    }

    LBMemoryModel model;
    model.build(stats);
    if (model.numDevices() == 0) return 1;

    int nb = 1;
    std::vector<std::vector<size_t>> used(1,
        std::vector<size_t>(model.numDevices(), 0));
    for (int i : moved) {
      int s = model.deviceOfPe(stats->from_proc[i]);
      int d = model.deviceOfPe(stats->to_proc[i]);
      size_t sig = model.stagedSize(i);
      if (s < 0 || d < 0 || s == d || sig == 0 ||
          LBMemoryModel::heldTransport(stats->from_proc[i], stats->to_proc[i]))
        continue;  // stages nothing: batch 0
      size_t Ss = model.device(s).stagingFree, Sd = model.device(d).stagingFree;
      int k = -1;
      for (int b = 0; b < nb; b++)
        if (used[b][s] + sig <= Ss && used[b][d] + sig <= Sd) { k = b; break; }
      if (k < 0) {
        nb++;
        used.push_back(std::vector<size_t>(model.numDevices(), 0));
        k = nb - 1;
      }
      used[k][s] += sig;
      used[k][d] += sig;
      batchOf[i] = k;
    }
    return nb;
  }
};

#endif  // LB_MEMORY_CONTRACT_H

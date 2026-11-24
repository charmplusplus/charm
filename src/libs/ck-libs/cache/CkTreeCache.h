#ifndef __CKTREECACHE_H__
#define __CKTREECACHE_H__

/** @file CkTreeCache.h
 *
 *  CkTreeCacheManager: a PROCESS-SHARED replacement for CkCacheManager's
 *  store, for caches whose entries are the nodes of a tree (ChaNGa's node
 *  cache). It keeps CkCacheManager's request/reply protocol -- the same
 *  CkCacheEntryType (request / unpack / free), the same
 *  CkCacheRequestorData callbacks, the same chunk lifecycle (cacheSync,
 *  finishedChunk) and the same member names the application calls -- but
 *  its STORE is one per process: the group's branches on the PEs of a
 *  process share a single lock-free store (owned by the rank-0 branch),
 *  so a remote node fetched once is seen by every PE of the process and a
 *  walker that reaches a fetched node follows a pointer instead of doing
 *  a per-PE map lookup. It stays a per-PE GROUP (not a nodegroup) because
 *  the application's delivery callbacks may be [local] entry methods of
 *  the requesting array element (ChaNGa's are), which must run on that
 *  element's PE: a reply is unpacked and installed on the PE it arrives
 *  at, and requestors parked from other PEs are resumed by a message to
 *  their own branch (deliverRemote).
 *
 *  The store is TreeCacheCore<Traits> (paratreet2's tree-cache core; the
 *  TreeCacheCore.h beside this file is a verbatim copy whose canonical
 *  source is paratreet2/treecache/TreeCacheCore.h) over the
 *  application's OWN process-shared tree, which
 *  the application exposes through Traits::root(). Missing children of
 *  that tree are represented by PLACEHOLDER nodes the manager creates in
 *  the empty child slots; a walker that must wait for a node parks its
 *  requestor on the placeholder; when the reply (a partial subtree, the
 *  "cache line") arrives, the manager builds it fully, then publishes it
 *  with one atomic child-pointer exchange in place of the placeholder and
 *  delivers the parked requestors exactly once. Concurrent readers see
 *  the placeholder or the complete line, never a partial state.
 *
 *  NO LOCKS ON THE MISS OR FILL PATH. Everything a walker or a reply
 *  touches is atomics on the tree (the core) or a PE-PRIVATE record: each
 *  PE appends its parked requestors, the placeholders it created and the
 *  lines it installed to its own lists; a parked opaque is (PE, index);
 *  resuming a requestor of another PE sends that PE the indices, and it
 *  reads only its own list. Teardown of a chunk runs when every chare of
 *  the process has finished it -- a quiescent point -- and then walks all
 *  PEs' lists. The one mutex (cacheSync) is taken once per chare per
 *  phase, at the phase boundary, never per node.
 *
 *  ONE REPLY PER REQUEST, DELIVERED AT ITS OWN PLACEHOLDER. A placeholder
 *  whose request latch is set (a request is in flight for it) is never
 *  replaced by another line that happens to contain its key: such a line
 *  ADOPTS the placeholder as its cut child and drops its own copy of that
 *  subtree. So every request's reply finds its placeholder still linked,
 *  and a walker that sent a request is resumed by that reply -- which is
 *  what lets the chunk be torn down as soon as every chare has finished,
 *  exactly as CkCacheManager does (a reply after teardown cannot occur).
 *  Placeholders nobody has requested (the intermediate chain a deep
 *  request creates above itself) ARE replaced by a line containing them;
 *  the line claims their latch first, so no request is sent for them
 *  afterwards. A closed placeholder always carries its replacement, so a
 *  late parker gets the installed node directly.
 *
 *  TRAITS CONTRACT (static inline functions over the application's node;
 *  no virtuals or function pointers), in addition to TreeCacheCore's
 *  four (key, parent, exchangeChild, parkedHead):
 *
 *    using Node = ...; using Key = ...;      // root key 1, child i of k = k*B+i
 *    static Node* root();                    // this phase's process-shared tree
 *    static int   branchFactor();            // B
 *    static Node* rawChild(const Node*, int which);           // atomic load
 *    static bool  casChild(Node* parent, int which, Node*& expected, Node* desired);
 *    static void  wireChild(Node* parent, int which, Node* child); // unpublished store
 *    static void  setParent(Node*, Node* parent);
 *    static bool  canHaveChildren(const Node*); // the walker descends below it
 *    static bool  isPlaceholder(const Node*);
 *    static Node* makePlaceholder(Key, Node* parent, int chunk);
 *    static void  freePlaceholder(Node*);
 *    static int   placeholderChunk(const Node* placeholder);
 *    static std::atomic<int>&   requestLatch(Node* placeholder); // 0 = not yet requested
 *    static std::atomic<Node*>& replacement(Node* placeholder);  // the node that took its place
 *
 *  The application's CkCacheEntryType::unpack(msg, chunk, home) returns the
 *  TOP node of the line with the line's own child pointers already
 *  resolved (ChaNGa: unpackNodes + retyping) and the cut children NULL;
 *  the manager does all linking into the shared tree. free() is never
 *  called by this manager: the line's message is freed once per line by
 *  the manager at chunk teardown (CkFreeMsg), so unpack must NOT take
 *  per-node references on it.
 */

#include "CkCache.h"
#include "TreeCacheCore.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

/// What getCache() returns: enough for the application's size() statistic.
class CkTreeCacheView {
public:
  size_t installed = 0;
  size_t size() const { return installed; }
};

template<class CkCacheKey, class Traits>
class CkTreeCacheManager : public CBase_CkTreeCacheManager<CkCacheKey, Traits> {
  using Node = typename Traits::Node;
  using Core = TreeCacheCore<Traits>;
  using Requestor = CkCacheRequestorData<CkCacheKey>;
  using Self = CkTreeCacheManager<CkCacheKey, Traits>;

  struct Install { Node* top; CkCacheFillMsg<CkCacheKey>* msg; };
  // Records of one PE for one chunk. Appended ONLY by that PE (no lock);
  // read by other PEs only at chunk teardown, a quiescent point.
  struct ChunkRecords {
    std::vector<Requestor> requestors;   ///< opaque = (rank, index)
    std::vector<Node*> placeholders;
    std::vector<Install> installs;
    std::map<CkCacheKey, Node*> inflight; ///< requests this PE sent: key -> placeholder
  };
  struct PeRecords {
    std::vector<ChunkRecords> chunks;
    int seenGen = -1;                    ///< last phase in which this PE freed its leftovers
#ifdef CKTREECACHE_TIMERS
    double tMiss = 0, tRecv = 0, tTeardown = 0, tFree = 0;  ///< wall time: miss path, reply path, unlink, free
#endif
  };
  static uint64_t makeOpaque(int rank, size_t index) { return ((uint64_t)rank << 40) | (uint64_t)index; }
  static int opaqueRank(uint64_t o) { return (int)(o >> 40); }
  static size_t opaqueIndex(uint64_t o) { return (size_t)(o & ((1ull << 40) - 1)); }
  /// A delivery to dispatch once the reply's bookkeeping is complete.
  struct Delivery { CkCacheKey key; Node* data; std::vector<uint64_t> opaques; };

  /// The process-shared state: owned by the rank-0 branch, reached by
  /// every other branch of the process through S().
  struct Shared {
    Core core;
    CkCacheEntryType<CkCacheKey>* type = nullptr;  ///< the single entry type served
    std::mutex phase_lock;               ///< the ONE mutex: cacheSync's phase setup, once per chare per phase
    std::atomic<int> phaseActive{0};
    std::atomic<int> phaseGen{0};        ///< incremented at every phase start
    int numChunks = 0;
    std::atomic<int> finishedChunks{0};
    std::vector<std::atomic<int>> chunkAck;
    CkCacheArrayCounter localChares;     ///< chares of the whole PROCESS this phase
    std::vector<PeRecords> pe;           ///< [rank] private records
    std::atomic<CmiUInt8> nRequests{0}, nMisses{0}, nLines{0}, nDelivered{0};
#if COSMO_STATS > 0 || defined(CKCACHE_STATS)
    std::atomic<CmiUInt8> nPlaceholders{0}, nLatchContended{0}, nAlreadyInstalled{0},
                          nAdopted{0}, nClaimed{0}, nRemoteDeliveries{0};
#endif
    std::atomic<long> installed{0};
    CkTreeCacheView view;
  };

  Shared* sh = nullptr;
  CkGroupID locMgr;                      ///< location manager of the requesting array

  ChunkRecords& mine(int chunk) { return S().pe[CkMyRank()].chunks[chunk]; }

  /// Bind this branch to the process's shared state (lazily: branch
  /// construction order within a process is not defined).
  Shared& S() {
    if (sh == nullptr) {
      Self* owner = (Self*)CkLocalBranchOther(this->thisgroup, 0);
      CkAssert(owner != nullptr && owner->sh != nullptr);
      sh = owner->sh;
    }
    return *sh;
  }

public:
  /// locMgr[0]: the location manager group of the array whose elements
  /// request data (its branches, one per PE, give the process-wide chare
  /// count per phase). Passed as an ID, like CkCacheManager: at group
  /// construction on a remote process the array may not exist yet, so
  /// nothing may be dereferenced here. (Array form: a plain CkGroupID
  /// constructor parameter would collide with the proxy's own CkGroupID
  /// constructor.)
  CkTreeCacheManager(int n, CkGroupID* locMgrs) : locMgr(locMgrs[0]) {
    CkAssert(n == 1);
    if (CkMyRank() == 0) sh = new Shared();
  }
  CkTreeCacheManager(CkMigrateMessage* m) : CBase_CkTreeCacheManager<CkCacheKey, Traits>(m) {}
  ~CkTreeCacheManager() { if (CkMyRank() == 0) delete sh; }
  /// Checkpoint/restart: between phases the store is empty, so only the
  /// binding to the requesting array persists.
  void pup(PUP::er& p) {
    CBase_CkTreeCacheManager<CkCacheKey, Traits>::pup(p);
    p | locMgr;
    if (p.isUnpacking() && CkMyRank() == 0) sh = new Shared();
  }

  // ------------------------------------------------------------------
  // Lookup: descend the shared tree from the root by the key's digits.
  // create = make placeholders in empty slots along the way (request
  // path); otherwise stop at the first empty slot (returns NULL).
  // ------------------------------------------------------------------
  Node* descend(CkCacheKey key, int chunk, bool create) {
    const int B = Traits::branchFactor();
    CkCacheKey path[128];
    int n = 0;
    for (CkCacheKey k = key; k > CkCacheKey(1); k /= B) path[n++] = k;
    Node* node = S().core.root;
    CkAssert(node != nullptr);
    for (int j = n - 1; j >= 0; j--) {
      const CkCacheKey ck = path[j];
      const int which = (int)(ck % B);
      if (!Traits::canHaveChildren(node)) {
        if (!create) return nullptr;
        CkAbort("CkTreeCacheManager: request for a key below a leaf of the shared tree");
      }
      Node* child = Traits::rawChild(node, which);
      if (child == nullptr) {
        if (!create) return nullptr;
        Node* ph = Traits::makePlaceholder(ck, node, chunk);
        Node* expected = nullptr;
        if (Traits::casChild(node, which, expected, ph)) {
          recordPlaceholder(chunk, ph);
          child = ph;
        } else {
          Traits::freePlaceholder(ph);
          child = expected;              // another PE's placeholder (or install) won
        }
      }
      node = child;
    }
    return node;
  }

  // ------------------------------------------------------------------
  // CkCacheManager-compatible API
  // ------------------------------------------------------------------

  /// Hit: the node. Miss: NULL, the requestor is parked and (once per
  /// process) the entry type's request() is issued. This form descends
  /// from the root by key (the walker knows only the key: ChaNGa's
  /// prefetch chunk roots); the walker's normal miss path is
  /// requestDataAt, which starts at the slot it stands at.
  void* requestData(CkCacheKey what, CkArrayIndex& toWhom, int chunk,
                    CkCacheEntryType<CkCacheKey>* t, Requestor& req) {
    Shared& s = S();
    CkAssert(chunk >= 0 && chunk < s.numChunks && s.chunkAck[chunk].load() > 0);
#ifdef CKTREECACHE_TIMERS
    const double t0 = CmiWallTimer();
#endif
    Node* node = descend(what, chunk, true);
    void* r = missOrHit(node, what, toWhom, chunk, t, req);
#ifdef CKTREECACHE_TIMERS
    s.pe[CkMyRank()].tMiss += CmiWallTimer() - t0;
#endif
    return r;
  }

  /// The walker's miss path: `parent`'s child `which` (key `what`) read as
  /// absent. One slot load, at most one CAS to plant the placeholder,
  /// then latch and park -- no descent.
  void* requestDataAt(Node* parent, int which, CkCacheKey what, CkArrayIndex& toWhom,
                      int chunk, CkCacheEntryType<CkCacheKey>* t, Requestor& req) {
    Shared& s = S();
    CkAssert(chunk >= 0 && chunk < s.numChunks && s.chunkAck[chunk].load() > 0);
#ifdef CKTREECACHE_TIMERS
    const double t0 = CmiWallTimer();
#endif
    Node* node = Traits::rawChild(parent, which);
    if (node == nullptr) {
      Node* ph = Traits::makePlaceholder(what, parent, chunk);
      Node* expected = nullptr;
      if (Traits::casChild(parent, which, expected, ph)) {
        recordPlaceholder(chunk, ph);
        node = ph;
      } else {
        Traits::freePlaceholder(ph);
        node = expected;
      }
    }
    CkAssert(Traits::key(node) == what);
    void* r = missOrHit(node, what, toWhom, chunk, t, req);
#ifdef CKTREECACHE_TIMERS
    s.pe[CkMyRank()].tMiss += CmiWallTimer() - t0;
#endif
    return r;
  }

private:
  void* missOrHit(Node* node, CkCacheKey what, CkArrayIndex& toWhom, int chunk,
                  CkCacheEntryType<CkCacheKey>* t, Requestor& req) {
    Shared& s = S();
    // One entry type serves the whole cache; requestors may each own an
    // instance of it (ChaNGa's TreePieces do), so keep the first one and
    // require nothing of the rest.
    if (s.type == nullptr) s.type = t;
    s.nRequests.fetch_add(1, std::memory_order_relaxed);
    if (!Traits::isPlaceholder(node)) return node;
    s.nMisses.fetch_add(1, std::memory_order_relaxed);
    // Latch BEFORE parking: a line that contains this key claims the
    // latch of an unrequested placeholder before replacing it, so once we
    // hold the latch the placeholder stays linked and our reply will find
    // it; if the line got there first, the latch tells us not to send.
    if (Traits::requestLatch(node).exchange(1) == 0) {
      mine(chunk).inflight[what] = node; // the reply comes back to this PE
      CkArrayIndex home(toWhom);
      void* d = s.type->request(home, what);
      CkAssert(d == nullptr);            // node caches reply by message
    }
#if COSMO_STATS > 0 || defined(CKCACHE_STATS)
    else s.nLatchContended.fetch_add(1, std::memory_order_relaxed);
#endif
    const uint64_t opaque = storeRequestor(chunk, req);
    if (s.core.park(node, opaque) == Core::ParkResult::AlreadyInstalled) {
#if COSMO_STATS > 0 || defined(CKCACHE_STATS)
      s.nAlreadyInstalled.fetch_add(1, std::memory_order_relaxed);
#endif
      // The install won the race: a closed placeholder carries the node
      // that replaced it. Deliver synchronously by returning it, as
      // CkCacheManager does for a hit.
      Node* installed = Traits::replacement(node).load();
      CkAssert(installed != nullptr && !Traits::isPlaceholder(installed));
      return installed;
    }
    return nullptr;
  }

public:
  void* requestDataNoFetch(CkCacheKey key, int chunk) {
    Node* node = descend(key, chunk, false);
    return (node != nullptr && !Traits::isPlaceholder(node)) ? node : nullptr;
  }

  /// The reply: a cache line for msg->key. Entry method; runs on the PE
  /// that sent the request (replyTo), concurrently with walkers on the
  /// others; that PE's own records name the placeholder, so no descent.
  void recvData(CkCacheFillMsg<CkCacheKey>* msg) {
    Shared& s = S();
#ifdef CKTREECACHE_TIMERS
    const double t0 = CmiWallTimer();
#endif
    const CkCacheKey key = msg->key;
    Node* slot = nullptr;
    int chunk = -1;
    for (int c = 0; c < s.numChunks && slot == nullptr; c++) {
      auto& inflight = mine(c).inflight;
      auto it = inflight.find(key);
      if (it != inflight.end()) { slot = it->second; chunk = c; inflight.erase(it); }
    }
    if (slot == nullptr) CkAbort("CkTreeCacheManager: reply for a key this PE did not request");
    CkAssert(Traits::isPlaceholder(slot) && Traits::placeholderChunk(slot) == chunk);
    CkArrayIndex home;                   // unused by node unpack
    Node* top = (Node*)s.type->unpack(msg, chunk, home);
    CkAssert(top != nullptr && Traits::key(top) == key);

    // Wire the line against whatever already hangs below the placeholder
    // (a placeholder chain from a deeper request, possibly with installed
    // nodes or requested placeholders along it), then publish the top.
    std::vector<Node*> claimed, adopted;
    wire(top, slot, chunk, claimed, adopted);
    recordInstall(chunk, top, msg);
    s.nLines.fetch_add(1, std::memory_order_relaxed);

    // Publication, drains and fix-ups first; the resumptions are
    // dispatched LAST, so that by the time a resumed chare can declare the
    // chunk finished (which may tear it down), this reply's bookkeeping
    // is complete.
    std::vector<Delivery> out;
    out.push_back(Delivery{key, top, {}});
    publish(slot, top, chunk, key, out.back().opaques);
    for (Node* ph : claimed) {           // interior waiters, after publication
      out.push_back(Delivery{Traits::key(ph), Traits::replacement(ph).load(), {}});
      Core::drainParked(ph, out.back().opaques);
    }
    // An adopted (requested) placeholder may meanwhile have been replaced
    // by its own reply, which then found our line's private slot too late:
    // finish that publication here (see publish()).
    for (Node* ph : adopted) {
      Node* rep = Traits::replacement(ph).load();
      if (rep != nullptr) {
        Node* p = Traits::parent(ph);
        const int which = (int)(Traits::key(ph) % Traits::branchFactor());
        Node* expected = ph;
        Traits::casChild(p, which, expected, rep);
      }
    }
#if COSMO_STATS > 0 || defined(CKCACHE_STATS)
    s.nClaimed.fetch_add(claimed.size(), std::memory_order_relaxed);
    s.nAdopted.fetch_add(adopted.size(), std::memory_order_relaxed);
#endif
#ifdef CKTREECACHE_TIMERS
    s.pe[CkMyRank()].tRecv += CmiWallTimer() - t0;
#endif
    for (auto& d : out) deliver(chunk, d.key, d.data, d.opaques);
  }

  /// Phase start, called by every local chare. The first caller of a
  /// phase sizes the chunks and counts the chares of the WHOLE PROCESS
  /// (every PE's location manager), which is the chunk-teardown target.
  void cacheSync(int& _numChunks, CkArrayIndex& chareIdx, int& localIdx) {
    Shared& s = S();
    std::lock_guard<std::mutex> g(s.phase_lock);
    if (s.phaseActive.load() == 0) {
      s.finishedChunks.store(0);
      s.localChares.reset();
      const int ranks = CkNodeSize(CkMyNode());
      for (int r = 0; r < ranks; r++) {
        CkLocMgr* mgr = (CkLocMgr*)CkLocalBranchOther(locMgr, r);
        mgr->iterate(s.localChares);
      }
      if (s.numChunks != _numChunks) {
        s.numChunks = _numChunks;
        std::vector<std::atomic<int>> fresh(s.numChunks);
        s.chunkAck.swap(fresh);
        // Records only grow: a PE frees its own leftovers itself (below),
        // so nothing here may touch another PE's records.
        if ((int)s.pe.size() != ranks) s.pe.resize(ranks);
        for (auto& pr : s.pe)
          if ((int)pr.chunks.size() < s.numChunks) pr.chunks.resize(s.numChunks);
      }
      for (int i = 0; i < s.numChunks; i++) s.chunkAck[i].store(s.localChares.count);
#if COSMO_STATS > 0 || defined(CKCACHE_STATS)
      // Per-phase counters, like CkCacheManager's.
      s.nRequests = s.nMisses = s.nLines = s.nDelivered = 0;
      s.nPlaceholders = s.nLatchContended = s.nAlreadyInstalled = s.nAdopted = s.nClaimed = s.nRemoteDeliveries = 0;
#endif
      s.core.root = Traits::root();
      s.core.branch_factor = Traits::branchFactor();
      CkAssert(s.core.root != nullptr);
      s.phaseGen.fetch_add(1);
      s.phaseActive.store(1);
    } else {
      _numChunks = s.numChunks;
    }
    localIdx = s.localChares.registered.get(chareIdx);
    CkAssert(localIdx != 0);
    // Once per PE per phase, before this PE's first request: a
    // freeChunkRecords message from the previous phase may not have been
    // processed yet on this PE, so free the leftovers now.
    PeRecords& pr = s.pe[CkMyRank()];
    const int gen = s.phaseGen.load();
    if (pr.seenGen != gen) {
      pr.seenGen = gen;
      for (size_t i = 0; i < pr.chunks.size(); i++) {
        ChunkRecords& c = pr.chunks[i];
        if (!c.installs.empty() || !c.placeholders.empty()) freeChunkRecords((int)i);
      }
    }
  }

  void writebackChunk(int) {}            // node caches are read-only

  /// Called by every chare of the process for every chunk; the last call
  /// for a chunk tears it down. No reply can arrive after this (see the
  /// header): every request's reply resumed its requestor. The teardown
  /// has two parts: UNLINKING every install and placeholder of the chunk
  /// from the shared tree (which must be clean before the application
  /// deletes it) -- atomics only, done here by the last caller -- and
  /// FREEING them, which touches nothing shared and is done by each PE
  /// for its own records, in parallel, on a message (freeChunkRecords).
  void finishedChunk(int chunk, CmiUInt8 /*weight*/) {
    Shared& s = S();
    CkAssert(chunk >= 0 && chunk < s.numChunks && s.chunkAck[chunk].load() > 0);
    if (s.chunkAck[chunk].fetch_sub(1) != 1) return;
#ifdef CKTREECACHE_TIMERS
    const double t0 = CmiWallTimer();
#endif
    // Quiescent: every chare of the process has finished this chunk, so
    // no PE is appending to its records for it or walking its nodes.
    const int B = Traits::branchFactor();
    for (auto& pr : s.pe) {
      ChunkRecords& c = pr.chunks[chunk];
      for (auto& in : c.installs) {
        Node* p = Traits::parent(in.top);
        if (p) {
          const int which = (int)(Traits::key(in.top) % B);
          if (Traits::rawChild(p, which) == in.top) Traits::exchangeChild(p, which, nullptr);
        }
      }
      for (Node* ph : c.placeholders) {
        Node* p = Traits::parent(ph);
        if (p) {
          const int which = (int)(Traits::key(ph) % B);
          if (Traits::rawChild(p, which) == ph) Traits::exchangeChild(p, which, nullptr);
        }
      }
    }
    const int firstPe = CkNodeFirst(CkMyNode());
    for (int r = 0; r < (int)s.pe.size(); r++) {
      if (r == CkMyRank()) freeChunkRecords(chunk);
      else this->thisProxy[firstPe + r].freeChunkRecords(chunk);
    }
#ifdef CKTREECACHE_TIMERS
    s.pe[CkMyRank()].tTeardown += CmiWallTimer() - t0;
#endif
    if (s.finishedChunks.fetch_add(1) + 1 == s.numChunks) {
      // The entry type object belongs to a requestor (ChaNGa: a member of
      // a TreePiece), which may migrate between phases: never keep it
      // across a phase.
      s.type = nullptr;
      s.phaseActive.store(0);
    }
  }

  /// Per-PE contribution; the rank-0 branch reports the process's numbers.
  void collectStatistics(const CkCallback& cb) {
    Shared& s = S();
    const bool owner = (CkMyRank() == 0);
#if COSMO_STATS > 0 || defined(CKCACHE_STATS)
    if (owner) {
      double tm = 0, tr = 0, tt = 0, tf = 0;
#ifdef CKTREECACHE_TIMERS
      for (auto& pr : s.pe) { tm += pr.tMiss; tr += pr.tRecv; tt += pr.tTeardown; tf += pr.tFree; }
#endif
      CkPrintf("[node %d] CkTreeCache: requests %llu misses %llu lines(msgs) %llu placeholders %llu "
               "latch_contended %llu already_installed %llu adopted %llu claimed %llu "
               "remote_deliveries %llu delivered %llu | PE-summed s: miss %.3f recv %.3f unlink %.3f free %.3f\n",
               CkMyNode(), (unsigned long long)s.nRequests.load(), (unsigned long long)s.nMisses.load(),
               (unsigned long long)s.nLines.load(), (unsigned long long)s.nPlaceholders.load(),
               (unsigned long long)s.nLatchContended.load(), (unsigned long long)s.nAlreadyInstalled.load(),
               (unsigned long long)s.nAdopted.load(), (unsigned long long)s.nClaimed.load(),
               (unsigned long long)s.nRemoteDeliveries.load(), (unsigned long long)s.nDelivered.load(),
               tm, tr, tt, tf);
    }
#endif
    CkCacheStatistics cs(owner ? s.nLines.load() : 0, owner ? s.nLines.load() : 0,
                         owner ? s.nMisses.load() : 0, 0, 0,
                         owner ? s.nRequests.load() : 0,
                         owner ? (CmiUInt8)s.installed.load() : 0, CkMyPe());
    this->contribute(sizeof(CkCacheStatistics), &cs, CkCacheStatistics::sum, cb);
  }

  CkTreeCacheView* getCache() {
    Shared& s = S();
    s.view.installed = (size_t)s.installed.load();
    return &s.view;
  }

  /// Entry method (also called directly by the last finisher for its own
  /// rank): free THIS PE's records of a torn-down chunk. Idempotent, and
  /// safe whenever it runs: the records were unlinked from the tree
  /// before the message was sent, and a late arrival is handled by the
  /// next cacheSync freeing leftovers itself.
  void freeChunkRecords(int chunk) {
    Shared& s = S();
#ifdef CKTREECACHE_TIMERS
    const double t0 = CmiWallTimer();
#endif
    ChunkRecords& c = mine(chunk);
    for (auto& in : c.installs) CkFreeMsg(in.msg);
    for (Node* ph : c.placeholders) Traits::freePlaceholder(ph);
    s.installed.fetch_sub((long)c.installs.size());
    CkAssert(c.inflight.empty());
    c.installs.clear();
    c.placeholders.clear();
    c.requestors.clear();
#ifdef CKTREECACHE_TIMERS
    s.pe[CkMyRank()].tFree += CmiWallTimer() - t0;
#endif
  }

  /// Entry method: resume requestors parked from THIS PE (their delivery
  /// callbacks may be [local] entries of the requesting elements).
  void deliverRemote(int chunk, CkCacheKey key, CmiUInt8 data, int n, CmiUInt8* opaques) {
    Shared& s = S();
    // Only this PE's records are read. A resumed walker may append to
    // them (a new miss) before the loop ends: copy the record first.
    for (int i = 0; i < n; i++) {
      CkAssert(opaqueRank(opaques[i]) == CkMyRank());
      Requestor req(mine(chunk).requestors[opaqueIndex(opaques[i])]);
      req.deliver(key, (void*)(uintptr_t)data, chunk);
      s.nDelivered.fetch_add(1, std::memory_order_relaxed);
    }
  }

private:
  // ------------------------------------------------------------------
  // Wiring a line against the existing subtree below its placeholder.
  // line = a node of the arriving line; existing = the node currently
  // at that key in the shared tree (the placeholder for the top, a
  // chain placeholder below it, or NULL). The line's child pointers are
  // its own (unpublished) memory, so plain stores are correct here; the
  // shared tree is only touched by publish() afterwards.
  //   claimed: unrequested placeholders the line replaces (drained after
  //            publication, their waiters get the line's node)
  //   adopted: requested placeholders kept as the line's cut children
  // ------------------------------------------------------------------
  void wire(Node* line, Node* existing, int chunk,
            std::vector<Node*>& claimed, std::vector<Node*>& adopted) {
    if (!Traits::canHaveChildren(line)) return;
    const int B = Traits::branchFactor();
    for (int i = 0; i < B; i++) {
      Node* lc = Traits::rawChild(line, i);
      Node* ec = existing ? Traits::rawChild(existing, i) : nullptr;
      if (ec != nullptr && Traits::isPlaceholder(ec)) {
        if (lc != nullptr && Traits::requestLatch(ec).exchange(1) == 0) {
          // Nobody requested it: the line's node takes its place.
          Traits::replacement(ec).store(lc);
          claimed.push_back(ec);
          Traits::setParent(lc, line);
          wire(lc, ec, chunk, claimed, adopted);
        } else {
          // A request is in flight (or the line has no data for it): keep
          // the placeholder; its own reply installs there.
          Traits::wireChild(line, i, ec);
          Traits::setParent(ec, line);
          if (lc != nullptr) adopted.push_back(ec);
        }
      } else if (ec != nullptr) {
        // Installed earlier by a deeper request: adopt it, drop the
        // line's copy (its memory stays inside the message).
        Traits::wireChild(line, i, ec);
        Traits::setParent(ec, line);
      } else if (lc != nullptr) {
        Traits::setParent(lc, line);
        wire(lc, nullptr, chunk, claimed, adopted);
      } else {
        Node* ph = Traits::makePlaceholder(Traits::key(line) * B + i, line, chunk);
        recordPlaceholder(chunk, ph);
        Traits::wireChild(line, i, ph);
      }
    }
  }

  /// Publish `top` in place of its placeholder `slot` and resume the
  /// waiters. The placeholder may at this moment be held in the PRIVATE
  /// slot of another line that adopted it and has not published yet (its
  /// parent pointer then still names the old parent): the CAS below
  /// lands in the old parent, and that line finishes the job after its
  /// own publication by reading the replacement we record first.
  void publish(Node* slot, Node* top, int chunk, CkCacheKey key, std::vector<uint64_t>& drained) {
    Traits::replacement(slot).store(top);
    const int which = (int)(key % Traits::branchFactor());
    Node* p = Traits::parent(slot);
    Node* expected = slot;
    Traits::casChild(p, which, expected, top);
    // If the placeholder is (still or again) reachable from the root,
    // its parent was re-linked meanwhile: publish there too.
    for (int guard = 0; guard < 64; guard++) {
      Node* now = descend(key, chunk, false);
      if (now != slot) break;
      p = Traits::parent(slot);
      expected = slot;
      Traits::casChild(p, which, expected, top);
    }
    Core::drainParked(slot, drained);
  }

  uint64_t storeRequestor(int chunk, Requestor& req) {
    ChunkRecords& c = mine(chunk);
    c.requestors.push_back(req);
    return makeOpaque(CkMyRank(), c.requestors.size() - 1);
  }
  void recordPlaceholder(int chunk, Node* ph) {
    if (chunk < 0) { CkAbort("CkTreeCacheManager: placeholder without a chunk"); }
    mine(chunk).placeholders.push_back(ph);
#if COSMO_STATS > 0 || defined(CKCACHE_STATS)
    S().nPlaceholders.fetch_add(1, std::memory_order_relaxed);
#endif
  }
  void recordInstall(int chunk, Node* top, CkCacheFillMsg<CkCacheKey>* msg) {
    mine(chunk).installs.push_back(Install{top, msg});
    S().installed.fetch_add(1);
  }
  /// Resume the drained requestors: those parked from this PE directly
  /// (the callback may be a [local] entry of a local element), the others
  /// through their own branch, which reads only its own records.
  void deliver(int chunk, CkCacheKey key, Node* data, const std::vector<uint64_t>& opaques) {
    if (opaques.empty()) return;
    Shared& s = S();
    const int myRank = CkMyRank();
    const int firstPe = CkNodeFirst(CkMyNode());
    std::vector<uint64_t> local;
    std::map<int, std::vector<CmiUInt8>> byRank;
    for (uint64_t o : opaques) {
      const int r = opaqueRank(o);
      if (r == myRank) local.push_back(o);
      else byRank[r].push_back((CmiUInt8)o);
    }
    for (auto& kv : byRank) {
#if COSMO_STATS > 0 || defined(CKCACHE_STATS)
      s.nRemoteDeliveries.fetch_add(1, std::memory_order_relaxed);
#endif
      this->thisProxy[firstPe + kv.first].deliverRemote(chunk, key, (CmiUInt8)(uintptr_t)data,
                                                        (int)kv.second.size(), kv.second.data());
    }
    for (uint64_t o : local) {
      Requestor req(mine(chunk).requestors[opaqueIndex(o)]);   // copy: the walker may append
      req.deliver(key, (void*)data, chunk);
      s.nDelivered.fetch_add(1, std::memory_order_relaxed);
    }
  }
};

#endif

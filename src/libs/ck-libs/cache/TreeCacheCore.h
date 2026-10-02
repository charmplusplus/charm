#ifndef PARATREET_TREECACHE_CORE_H_
#define PARATREET_TREECACHE_CORE_H_

// VENDORED COPY. The canonical source of this header is paratreet2:
//   github.com/UIUC-PPL/paratreet2, treecache/TreeCacheCore.h
//   (copied verbatim from commit f46e298, 2026-09-26).
// paratreet2 is its permanent home (the atomics-based SMP tree cache is
// a plain header with its own runtime-free test there); charm carries
// this copy only because CkTreeCache.h, the Charm++ wrapper and message
// orchestration around it, cannot depend on paratreet2. Do not edit it
// here: change it in paratreet2 and copy it over; `cmp` against the
// paratreet2 file must report no difference.

// TreeCacheCore: the node-type-INDEPENDENT part of the passive SMP tree
// cache (design/smp-cache-extraction.md, section 2). One instance is
// shared by every worker thread of a process. It owns exactly three
// things: the root of the process-shared tree, the placeholder park /
// install contract, and atomic publication of an installed node in place
// of its placeholder. Everything about what a node IS — its type, its
// allocation, its payload, how a partial subtree is built from a reply —
// belongs to the client (paratreet2's TreeCache<Data> below it; ChaNGa's
// binding over Tree::GenericTreeNode later) and reaches this class only
// through the Traits parameter.
//
// TRAITS CONTRACT. Every function is a static inline forwarder over the
// client's node; no virtuals, function pointers or std::function, so the
// instantiation compiles to the same code as direct member access.
//
//   struct Traits {
//     using Node = ...;             // the client's node type
//     using Key  = ...;             // unsigned integer; the ROOT has key 1
//                                   // and child i of key k has key
//                                   // k * branch_factor + i (SFC keys;
//                                   // ChaNGa's NodeKey follows the same
//                                   // rule)
//     static Key   key(const Node*);
//     static Node* parent(const Node*);
//     // Atomic child-pointer exchange on `parent`'s slot `which`; returns
//     // the displaced pointer. THE publication primitive.
//     static Node* exchangeChild(Node* parent, int which, Node* child);
//     // The node's parked-waiter list head. MUST be a field ON the node
//     // (not a side table): the park-vs-install race is closed against
//     // the node's own publication, and park is then one CAS.
//     static std::atomic<void*>& parkedHead(Node*);
//   };
//
// No runtime dependency: plain C++ over <atomic>. No messaging: how a
// missing node is REQUESTED, and how returned waiters are RESUMED, are the
// client's.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

template <typename Traits>
class TreeCacheCore {
public:
  using Node = typename Traits::Node;
  using Key = typename Traits::Key;

  // ---- park / install (phase 2 of the extraction design) ----
  // A walker that must wait for a node's data PARKS an opaque value on the
  // placeholder; install (the swapIn drain) returns every parked opaque
  // EXACTLY ONCE to the caller, which owns scheduling the resumptions. The
  // race between a late park and the install is closed here — against
  // install's atomic publication — by closing the list with a sentinel:
  // a park that finds the sentinel returns AlreadyInstalled and the caller
  // proceeds as if the data had been present (the lost-wakeup fix,
  // provided once here instead of re-solved by every client).
  struct ParkedEntry {
    uint64_t opaque;
    ParkedEntry* next;
  };
  static void* closedSentinel() {
    static char sentinel_storage;
    return (void*)&sentinel_storage;
  }
  enum class ParkResult { Parked, AlreadyInstalled };

  ParkResult park(Node* slot, uint64_t opaque) {
    std::atomic<void*>& parked_head = Traits::parkedHead(slot);
    // Consecutive-duplicate suppression, matching the old Resumer
    // waiting-list behavior: one walker sweeping many payloads against the
    // same placeholder parks once (a racing other-lane entry in between
    // just costs a duplicate resumption, which the traverser tolerates).
    void* head = parked_head.load();
    if (head != closedSentinel() && head != nullptr &&
        ((ParkedEntry*)head)->opaque == opaque)
      return ParkResult::Parked;
    auto* entry = new ParkedEntry{opaque, nullptr};
    while (true) {
      head = parked_head.load();
      if (head == closedSentinel()) {
        delete entry;
        return ParkResult::AlreadyInstalled;
      }
      entry->next = (ParkedEntry*)head;
      if (parked_head.compare_exchange_weak(head, (void*)entry))
        return ParkResult::Parked;
    }
  }

  // Only PLACEHOLDERS carry an open parked list. The client calls this on
  // every node it creates that is NOT a placeholder (local, boundary,
  // installed), so a park on such a node returns AlreadyInstalled instead
  // of accepting a waiter nobody would ever drain. (Found by the standalone
  // unit test; the library contract does not rely on walkers only parking
  // on placeholder types.)
  static void closeParkedList(Node* node) {
    Traits::parkedHead(node).store(closedSentinel());
  }

  // Atomically publish an installed node in place of its placeholder, and
  // collect the placeholder's parked waiters into `parked` (each opaque
  // handed back exactly once — the caller owns waking them). Waiters that
  // parked before the exchange are drained here; one that parks after sees
  // the closed list and gets AlreadyInstalled from park(), so no wakeup is
  // lost (the same guarantee the old requested-bitmask handoff provided;
  // see the fanout-fix record, design/walk-uf2-overlap.md step 2).
  // PRECONDITION: `to_swap` and its whole partial subtree are fully built
  // before this call — concurrent lookups see the placeholder or the
  // complete subtree, never a partial state.
  void swapIn(Node* to_swap, std::vector<uint64_t>& parked) {
    if (Traits::key(to_swap) > Key(1)) {
      auto which_child = Traits::key(to_swap) % branch_factor;
      Node* displaced = Traits::exchangeChild(Traits::parent(to_swap),
                                              (int)which_child, to_swap);
      if (displaced) drainParked(displaced, parked);
    }
    else {
      std::swap(root, to_swap);
      // to_swap now holds the displaced old root; NULL on the very first
      // swap (the starter pack installing the initial root).
      if (to_swap) drainParked(to_swap, parked);
    }
  }

  Node* root = nullptr;
  size_t branch_factor = 0;

  // Close a displaced placeholder's parked list and collect the opaques.
  // Exactly-once: the exchange with the sentinel wins against concurrent
  // parks (they either landed before — collected here — or see the
  // sentinel and self-handle). swapIn calls this for the placeholder it
  // displaces; a client that replaces INTERIOR placeholders while wiring
  // a partial subtree (a placeholder chain below the installed top, see
  // the ChaNGa binding) calls it directly for those, AFTER the top's
  // publication, so a late parker that re-descends finds the new node.
  static void drainParked(Node* displaced, std::vector<uint64_t>& out) {
    void* head = Traits::parkedHead(displaced).exchange(closedSentinel());
    if (head == closedSentinel()) return; // already drained
    auto* entry = (ParkedEntry*)head;
    while (entry) {
      out.push_back(entry->opaque);
      auto* next = entry->next;
      delete entry;
      entry = next;
    }
  }
};

#endif // PARATREET_TREECACHE_CORE_H_

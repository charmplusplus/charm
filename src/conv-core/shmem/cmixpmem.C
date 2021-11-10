#include "cmi-shmem-common.h"
#include <map>
#include <memory>
#include <vector>

extern "C" {
#include <xpmem.h>
}

// "borrowed" from VADER
// (https://github.com/open-mpi/ompi/tree/386ba164557bb8115131921041757be94a989646/opal/mca/smsc/xpmem)
#define OPAL_DOWN_ALIGN(x, a, t) ((x) & ~(((t)(a)-1)))
#define OPAL_DOWN_ALIGN_PTR(x, a, t) \
  ((t)OPAL_DOWN_ALIGN((uintptr_t)x, a, uintptr_t))
#define OPAL_ALIGN(x, a, t) (((x) + ((t)(a)-1)) & ~(((t)(a)-1)))
#define OPAL_ALIGN_PTR(x, a, t) ((t)OPAL_ALIGN((uintptr_t)x, a, uintptr_t))
#define OPAL_ALIGN_PAD_AMOUNT(x, s) \
  ((~((uintptr_t)(x)) + 1) & ((uintptr_t)(s)-1))

CpvStaticDeclare(int, handle_init);

struct init_msg_ {
  char core[CmiMsgHeaderSizeBytes];
  std::size_t key;
  int from;
  xpmem_segid_t segid;
  ipc_shared_* shared;
};

// NOTE ( we should eventually detach xpmem segments at close )
//      ( it's not urgently needed since xpmem does it for us )
struct CmiIpcManager : public ipc_metadata_ {
  // maps ranks to segments
  std::map<int, xpmem_segid_t> segments;
  // maps segments to xpmem apids
  std::map<xpmem_segid_t, xpmem_apid_t> instances;
  // number of physical peers
  int nPeers;
  // create our local shared data
  CmiIpcManager(std::size_t key) : ipc_metadata_(key) {
    this->shared[this->mine] = makeIpcShared_();
  }

  void put_segment(int proc, const xpmem_segid_t& segid) {
    auto ins = this->segments.emplace(proc, segid);
    CmiAssert(ins.second);
  }

  xpmem_segid_t get_segment(int proc) {
    auto search = this->segments.find(proc);
    if (search == std::end(this->segments)) {
      if (mine == proc) {
        auto segid =
            xpmem_make(0, XPMEM_MAXADDR_SIZE, XPMEM_PERMIT_MODE, (void*)0666);
        this->put_segment(mine, segid);
        return segid;
      } else {
        return -1;
      }
    } else {
      return search->second;
    }
  }

  xpmem_apid_t get_instance(int proc) {
    auto segid = this->get_segment(proc);
    if (segid >= 0) {
      auto search = this->instances.find(segid);
      if (search == std::end(this->instances)) {
        auto apid = xpmem_get(segid, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
        CmiAssertMsg(apid >= 0, "invalid segid?");
        auto ins = this->instances.emplace(segid, apid);
        CmiAssert(ins.second);
        search = ins.first;
      }
      return search->second;
    } else {
      return -1;
    }
  }
};

void* translateAddr_(ipc_manager_ptr_& meta, int proc, void* remote_ptr,
                     const std::size_t& size) {
  if (proc == meta->mine) {
    return remote_ptr;
  } else {
    auto apid = meta->get_instance(proc);
    CmiAssert(apid >= 0);
    // this magic was borrowed from VADER
    uintptr_t attach_align = 1 << 23;
    auto base = OPAL_DOWN_ALIGN_PTR(remote_ptr, attach_align, uintptr_t);
    auto bound =
        OPAL_ALIGN_PTR(remote_ptr + size - 1, attach_align, uintptr_t) + 1;

    using offset_type = decltype(xpmem_addr::offset);
    xpmem_addr addr{.apid = apid, .offset = (offset_type)base};
    auto* ctx = xpmem_attach(addr, bound - base, NULL);
    CmiAssert(ctx != (void*)-1);

    return (void*)((uintptr_t)ctx +
                   (ptrdiff_t)((uintptr_t)remote_ptr - (uintptr_t)base));
  }
}

static void handleInitialize_(void* msg) {
  auto* imsg = (init_msg_*)msg;
  auto& meta = (CsvAccess(managers_))[(imsg->key - 1)];
  // extract the segment id and shared region
  // from the msg (registering it in our metadata)
  meta->put_segment(imsg->from, imsg->segid);
  meta->shared[imsg->from] = (ipc_shared_*)translateAddr_(
      meta, imsg->from, imsg->shared, sizeof(ipc_shared_));
  // then free the message
  CmiFree(imsg);
  // if we received messages from all our peers:
  if (meta->nPeers == meta->shared.size()) {
    // resume the sleeping thread
    if (CmiMyPe() == 0) {
      CmiPrintf("CMI> xpmem pool init'd with %luB segment.\n",
                CpvAccess(kSegmentSize));
    }

    awakenSleepers_();
  }
}

void CmiIpcInit(char** argv) {
  CsvInitialize(ipc_manager_map_, managers_);

  initSleepers_();
  initSegmentSize_(argv);

  CpvInitialize(int, handle_init);
  CpvAccess(handle_init) = CmiRegisterHandler(handleInitialize_);
}

CmiIpcManager* CmiMakeIpcManager(CthThread th) {
  putSleeper_(th);

#if CMK_SMP
  // ensure all sleepers are reg'd
  CmiNodeAllBarrier();
#endif

  CmiIpcManager* meta;
  if (CmiMyRank() == 0) {
    auto key = CsvAccess(managers_).size() + 1;
    meta = new CmiIpcManager(key);
    CsvAccess(managers_).emplace_back(meta);
  } else {
#if CMK_SMP
    // pause until the metadata is ready
    CmiNodeAllBarrier();
#endif
    return CsvAccess(managers_).back().get();
  }

  int* pes;
  int nPes;
  auto thisPe = CmiMyPe();
  auto thisNode = CmiPhysicalNodeID(CmiMyPe());
  CmiGetPesOnPhysicalNode(thisNode, &pes, &nPes);
  auto nSize = CmiMyNodeSize();
  auto nProcs = nPes / nSize;
  meta->nPeers = nProcs;

  if (nProcs > 1) {
    auto* imsg = (init_msg_*)CmiAlloc(sizeof(init_msg_));
    CmiSetHandler(imsg, CpvAccess(handle_init));
    imsg->key = meta->key;
    imsg->from = meta->mine;
    imsg->segid = meta->get_segment(meta->mine);
    imsg->shared = meta->shared[meta->mine];
    // send messages to all the pes on this node
    for (auto i = 0; i < nProcs; i++) {
      auto& pe = pes[i * nSize];
      auto last = i == (nProcs - 1);
      if (pe == thisPe) {
        if (last) {
          CmiFree(imsg);
        }
        continue;
      } else if (last) {
        // free'ing with the last send
        CmiSyncSendAndFree(pe, sizeof(init_msg_), (char*)imsg);
      } else {
        // then sending (without free) otherwise
        CmiSyncSend(pe, sizeof(init_msg_), (char*)imsg);
      }
    }
  } else {
    // single process -- wake up sleeping thread(s)
    awakenSleepers_();
  }

#if CMK_SMP
  // signal that the metadata is ready
  CmiNodeAllBarrier();
#endif

  return meta;
}

#include <cmi-shmem-internal.hh>
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

struct init_msg_ {
  char core[CmiMsgHeaderSizeBytes];
  int from;
  xpmem_segid_t segid;
  ipc_shared_* shared;
};

// TODO ( detach xpmem segments at close )
// ( not urgently needed since xpmem does it for us )
struct ipc_xpmem_metadata_ : public ipc_metadata_ {
  // maps ranks to segments
  std::map<int, xpmem_segid_t> segments;
  // maps segments to xpmem apids
  std::map<xpmem_segid_t, xpmem_apid_t> instances;
  // number of physical peers
  int nPeers;
  // create our local shared data
  ipc_xpmem_metadata_(void) { this->shared[this->mine] = makeIpcShared_(); }

  void put_segment(int rank, const xpmem_segid_t& segid) {
    auto ins = this->segments.emplace(rank, segid);
    CmiAssert(ins.second);
  }

  xpmem_segid_t get_segment(int rank) {
    auto search = this->segments.find(rank);
    if (search == std::end(this->segments)) {
      if (mine == rank) {
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

  xpmem_apid_t get_instance(int rank) {
    auto segid = this->get_segment(rank);
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

void* translateAddr_(ipc_xpmem_metadata_* meta, int rank, void* remote_ptr,
                     const std::size_t& size) {
  // TODO ( add support for SMP mode )
  auto mine = meta->mine;
  if (mine == rank) {
    return remote_ptr;
  } else {
    auto apid = meta->get_instance(rank);
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
  auto* meta = (ipc_xpmem_metadata_*)CsvAccess(metadata_).get();
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

void CmiIpcInitMetadata(char** argv, CthThread th) {
  initSleepers_();
  initSegmentSize_(argv);
  CmiNodeAllBarrier();

  putSleeper_(th);

#if CMK_SMP
  // ensure all sleepers are reg'd
  CmiNodeAllBarrier();
#endif

  ipc_xpmem_metadata_* meta;
  if (CmiMyRank() == 0) {
    meta = new ipc_xpmem_metadata_();
    CsvInitialize(ipc_metadata_ptr_, metadata_);
    CsvAccess(metadata_).reset(meta);
  } else {
    return;
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
    auto initHdl = CmiRegisterHandler(handleInitialize_);
    auto* imsg = (init_msg_*)CmiAlloc(sizeof(init_msg_));
    CmiSetHandler(imsg, initHdl);
    imsg->from = meta->mine;
    imsg->segid = meta->get_segment(meta->mine);
    imsg->shared = meta->shared[meta->mine];
    // send messages to all the procs on this node
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
}

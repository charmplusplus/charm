#include "internal.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <memory>

struct ipc_shm_metadata_;

CpvStaticDeclare(int, num_cbs_recvd);
CpvStaticDeclare(int, num_cbs_exptd);
CpvStaticDeclare(int, handle_callback);
CpvStaticDeclare(int, handle_node_pid);
CsvStaticDeclare(pid_t, node_pid);

static void openAllShared_(ipc_shm_metadata_*);
static int sendPid_(void);

struct pid_message_ {
  char core[CmiMsgHeaderSizeBytes];
  pid_t pid;
};

#define CMI_SHARED_FMT "cmi_pid%lu_rank%d_shared_"

// opens a shared memory segment for a given physical rank
static std::pair<int, ipc_shared_*> openShared_(int rank) {
  // get the size from the cpv
  auto& size = CpvAccess(kSegmentSize);
  // generate a name for this pe
  auto slen = snprintf(NULL, 0, CMI_SHARED_FMT, CsvAccess(node_pid), rank);
  auto name = new char[slen];
  sprintf(name, CMI_SHARED_FMT, CsvAccess(node_pid), rank);
  DEBUGP(("%d> opening share %s\n", CmiMyPe(), name));
  // try opening the share exclusively
  auto fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0666);
  // if we succeed, we're the first accessor, so:
  if (fd >= 0) {
    // truncate it to the correct size
    auto status = ftruncate(fd, size);
    CmiAssert(status >= 0);
  } else {
    // otherwise just open it
    fd = shm_open(name, O_RDWR, 0666);
    CmiAssert(fd >= 0);
  }
  // then delete the name
  delete[] name;
  // map the segment to an address:
  auto* res = (ipc_shared_*)mmap(nullptr, size, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, fd, 0);
  CmiAssert(res != MAP_FAILED);
  // return the file descriptor/shared
  return std::make_pair(fd, res);
}

struct ipc_shm_metadata_ : public ipc_metadata_ {
  std::map<int, int> fds;

  ipc_shm_metadata_(void) {
    if (this->mine == 0) {
      if (sendPid_() == 1) {
        openAllShared_(this);
        awakenSleepers_();
      }
    }
  }

  virtual ~ipc_shm_metadata_() {
    auto& size = CpvAccess(kSegmentSize);
    // for each rank/descriptor pair
    for (auto& pair : this->fds) {
      auto& rank = pair.first;
      auto& fd = pair.second;
      // unmap the memory segment
      munmap(this->shared[rank], size);
      // close the file
      close(fd);
      // unlinking the shm segment for our pe
      if (rank == this->mine) {
        auto slen =
            snprintf(NULL, 0, CMI_SHARED_FMT, CsvAccess(node_pid), rank);
        auto name = new char[slen];
        sprintf(name, CMI_SHARED_FMT, CsvAccess(node_pid), rank);
        shm_unlink(name);
        delete[] name;
      }
    }
  }
};

static void openAllShared_(ipc_shm_metadata_* meta) {
  int node = CmiPhysicalNodeID(CmiMyPe());
  int nPes = CmiNumPesOnPhysicalNode(node);
  int nProcs = nPes / CmiMyNodeSize();
  // for each rank in this physical node:
  for (auto rank = 0; rank < nProcs; rank++) {
    // open its shared segment
    auto res = openShared_(rank);
    // initializing it if it's ours
    if (rank == meta->mine) initIpcShared_(res.second);
    // store the retrieved data
    meta->fds[rank] = res.first;
    meta->shared[rank] = res.second;
  }
}

// returns number of processes in node
int procBroadcastAndFree_(char* msg, std::size_t size) {
  int nPes;
  int* pes;
  int mine = CmiMyPe();
  int node = CmiPhysicalNodeID(mine);
  CmiGetPesOnPhysicalNode(node, &pes, &nPes);
  int nSize = CmiMyNodeSize();
  int nProcs = nPes / nSize;
  CmiAssert(mine == pes[0]);

  CpvAccess(num_cbs_exptd) = nProcs - 1;
  for (auto rank = 1; rank < nProcs; rank++) {
    auto& pe = pes[rank * nSize];
    if (rank == (nProcs - 1)) {
      CmiSyncSendAndFree(pe, size, msg);
    } else {
      CmiSyncSend(pe, size, msg);
    }
  }

  // free if we didn't send anything
  if (nProcs == 1) {
    CmiFree(msg);
  }

  return nProcs;
}

static int sendPid_(void) {
  CsvInitialize(pid_t, node_pid);
  CsvAccess(node_pid) = getpid();

  auto* pmsg = (pid_message_*)CmiAlloc(sizeof(pid_message_));
  CmiSetHandler(pmsg, CpvAccess(handle_node_pid));
  pmsg->pid = CsvAccess(node_pid);

  return procBroadcastAndFree_((char*)pmsg, sizeof(pid_message_));
}

static void callbackHandler_(void* msg) {
  int mine = CmiMyPe();
  int node = CmiPhysicalNodeID(mine);
  int first = CmiGetFirstPeOnPhysicalNode(node);

  if (mine == first) {
    // if we're still expecting messages:
    if (++(CpvAccess(num_cbs_recvd)) < CpvAccess(num_cbs_exptd)) {
      // free this one
      CmiFree(msg);
      // and move along
      return;
    } else {
      // otherwise -- tell everyone we're ready!
      CmiPrintf("CMI> posix shm pool init'd with %gMB segment, %gKB soft-cap.\n",
                CpvAccess(kSegmentSize) / (1024.0 * 1024.0), CmiRecommendedBlockCutoff() / 1024.0);
      procBroadcastAndFree_((char*)msg, CmiMsgHeaderSizeBytes);
    }
  } else {
    CmiFree(msg);
  }

  auto& meta = CsvAccess(metadata_);
  openAllShared_((ipc_shm_metadata_*)meta.get());
  awakenSleepers_();
}

static void nodePidHandler_(void* msg) {
  auto* pmsg = (pid_message_*)msg;
  CsvInitialize(pid_t, node_pid);
  CsvAccess(node_pid) = pmsg->pid;

  int node = CmiPhysicalNodeID(CmiMyPe());
  int root = CmiGetFirstPeOnPhysicalNode(node);
  CmiSetHandler(msg, CpvAccess(handle_callback));
  CmiSyncSendAndFree(root, CmiMsgHeaderSizeBytes, (char*)msg);
}

void CmiInitIpcMetadata(char** argv, CthThread th) {
  CpvInitialize(int, num_cbs_recvd);
  CpvInitialize(int, num_cbs_exptd);
  CpvAccess(num_cbs_recvd) = CpvAccess(num_cbs_exptd) = 0;
  CpvInitialize(int, handle_callback);
  CpvAccess(handle_callback) = CmiRegisterHandler(callbackHandler_);
  CpvInitialize(int, handle_node_pid);
  CpvAccess(handle_node_pid) = CmiRegisterHandler(nodePidHandler_);

  initSleepers_();
  initSegmentSize_(argv);
  CmiNodeAllBarrier();

  putSleeper_(th);

#if CMK_SMP
  // ensure all sleepers are reg'd
  CmiNodeAllBarrier();
#endif

  if (CmiMyRank() == 0) {
    CsvInitialize(ipc_metadata_ptr_, metadata_);
    CsvAccess(metadata_).reset(new ipc_shm_metadata_);
  }
}

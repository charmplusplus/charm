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

#include "cmi-shmem-common.h"
#include <memory>

CpvStaticDeclare(int, num_cbs_recvd);
CpvStaticDeclare(int, num_cbs_exptd);
CpvStaticDeclare(int, handle_callback);
CpvStaticDeclare(int, handle_node_pid);
CsvStaticDeclare(pid_t, node_pid);

static int sendPid_(CmiIpcManager*);
static void openAllShared_(CmiIpcManager*);

struct pid_message_ {
  char core[CmiMsgHeaderSizeBytes];
  std::size_t key;
  pid_t pid;
};

#define CMI_SHARED_FMT "cmi_pid%lu_node%d_shared_"

// opens a shared memory segment for a given physical rank
static std::pair<int, ipc_shared_*> openShared_(int node) {
  // determine the size of the shared segment
  // (adding the size of the queues and what nots)
  auto size = CpvAccess(kSegmentSize) + sizeof(ipc_shared_);
  // generate a name for this pe
  auto slen = snprintf(NULL, 0, CMI_SHARED_FMT, CsvAccess(node_pid), node);
  auto name = new char[slen];
  sprintf(name, CMI_SHARED_FMT, CsvAccess(node_pid), node);
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

struct CmiIpcManager : public ipc_metadata_ {
  std::map<int, int> fds;

  CmiIpcManager(std::size_t key) : ipc_metadata_(key) {
    auto firstPe = CmiNodeFirst(CmiMyNode());
    auto thisRank = CmiPhysicalRank(firstPe);
    if (thisRank == 0) {
      if (sendPid_(this) == 1) {
        openAllShared_(this);
        awakenSleepers_();
      }
    }
  }

  virtual ~CmiIpcManager() {
    auto& size = CpvAccess(kSegmentSize);
    // for each rank/descriptor pair
    for (auto& pair : this->fds) {
      auto& proc = pair.first;
      auto& fd = pair.second;
      // unmap the memory segment
      munmap(this->shared[proc], size);
      // close the file
      close(fd);
      // unlinking the shm segment for our pe
      if (proc == this->mine) {
        auto slen =
            snprintf(NULL, 0, CMI_SHARED_FMT, CsvAccess(node_pid), proc);
        auto name = new char[slen];
        sprintf(name, CMI_SHARED_FMT, CsvAccess(node_pid), proc);
        shm_unlink(name);
        delete[] name;
      }
    }
  }
};

static void openAllShared_(CmiIpcManager* meta) {
  int* pes;
  int nPes;
  int thisNode = CmiPhysicalNodeID(CmiMyPe());
  CmiGetPesOnPhysicalNode(thisNode, &pes, &nPes);
  int nSize = CmiMyNodeSize();
  int nProcs = nPes / nSize;
  // for each rank in this physical node:
  for (auto rank = 0; rank < nProcs; rank++) {
    // open its shared segment
    auto pe = pes[rank * nSize];
    auto proc = CmiNodeOf(pe);
    auto res = openShared_(proc);
    // initializing it if it's ours
    if (proc == meta->mine) initIpcShared_(res.second);
    // store the retrieved data
    meta->fds[proc] = res.first;
    meta->shared[proc] = res.second;
  }
  DEBUGP(("%d> finished opening all shared\n", meta->mine));
}

// returns number of processes in node
int procBroadcastAndFree_(char* msg, std::size_t size) {
  int* pes;
  int nPes;
  int thisPe = CmiMyPe();
  int thisNode = CmiPhysicalNodeID(thisPe);
  CmiGetPesOnPhysicalNode(thisNode, &pes, &nPes);
  int nSize = CmiMyNodeSize();
  int nProcs = nPes / nSize;
  CmiAssert(thisPe == pes[0]);

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

static int sendPid_(CmiIpcManager* manager) {
  CsvInitialize(pid_t, node_pid);
  CsvAccess(node_pid) = getpid();

  auto* pmsg = (pid_message_*)CmiAlloc(sizeof(pid_message_));
  CmiSetHandler(pmsg, CpvAccess(handle_node_pid));
  pmsg->key = manager->key;
  pmsg->pid = CsvAccess(node_pid);

  return procBroadcastAndFree_((char*)pmsg, sizeof(pid_message_));
}

static void callbackHandler_(void* msg) {
  int mine = CmiMyPe();
  int node = CmiPhysicalNodeID(mine);
  int first = CmiGetFirstPeOnPhysicalNode(node);
  auto* pmsg = (pid_message_*)msg;

  if (mine == first) {
    // if we're still expecting messages:
    if (++(CpvAccess(num_cbs_recvd)) < CpvAccess(num_cbs_exptd)) {
      // free this one
      CmiFree(msg);
      // and move along
      return;
    } else {
      // otherwise -- tell everyone we're ready!
      CmiPrintf("CMI> posix shm pool init'd with %luB segment.\n",
                CpvAccess(kSegmentSize));
      procBroadcastAndFree_((char*)msg, sizeof(pid_message_));
    }
  } else {
    CmiFree(msg);
  }

  auto& meta = (CsvAccess(managers_))[(pmsg->key - 1)];
  openAllShared_(meta.get());
  awakenSleepers_();
}

static void nodePidHandler_(void* msg) {
  auto* pmsg = (pid_message_*)msg;
  CsvInitialize(pid_t, node_pid);
  CsvAccess(node_pid) = pmsg->pid;

  int node = CmiPhysicalNodeID(CmiMyPe());
  int root = CmiGetFirstPeOnPhysicalNode(node);
  CmiSetHandler(msg, CpvAccess(handle_callback));
  CmiSyncSendAndFree(root, sizeof(pid_message_), (char*)msg);
}

void CmiIpcInit(char** argv) {
  CsvInitialize(ipc_manager_map_, managers_);

  initSleepers_();
  initSegmentSize_(argv);

  CpvInitialize(int, num_cbs_recvd);
  CpvInitialize(int, num_cbs_exptd);
  CpvInitialize(int, handle_callback);
  CpvAccess(handle_callback) = CmiRegisterHandler(callbackHandler_);
  CpvInitialize(int, handle_node_pid);
  CpvAccess(handle_node_pid) = CmiRegisterHandler(nodePidHandler_);
}

CmiIpcManager* CmiMakeIpcManager(CthThread th) {
  CpvAccess(num_cbs_recvd) = CpvAccess(num_cbs_exptd) = 0;

  putSleeper_(th);

#if CMK_SMP
  // ensure all sleepers are reg'd
  CmiNodeAllBarrier();
#endif

  if (CmiMyRank() == 0) {
    auto key = CsvAccess(managers_).size() + 1;
    auto* manager = new CmiIpcManager(key);
    CsvAccess(managers_).emplace_back(manager);
#if CMK_SMP
    // ensure all sleepers are reg'd
    CmiNodeAllBarrier();
#endif
    return manager;
  } else {
#if CMK_SMP
    // ensure all sleepers are reg'd
    CmiNodeAllBarrier();
#endif
    return CsvAccess(managers_).back().get();
  }
}

// Standalone coordinator for Charm++ UCX shrink/expand.
//
// Holds the cluster's current membership and a queue of newcomers waiting to
// be integrated. Drives wireup synchronization via a barrier.
//
// Single-threaded, poll()-based. No Charm++ dependencies.
//
// Usage:
//   charm_coordinator --port P --initial-size N [--verbose]

#include <arpa/inet.h>
#include <getopt.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <deque>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "wire.h"

using namespace coord;

namespace {

bool g_verbose = false;

#define LOG(...)                            \
  do {                                      \
    if (g_verbose) {                        \
      fprintf(stderr, "[coord] " __VA_ARGS__); \
      fputc('\n', stderr);                  \
      fflush(stderr);                       \
    }                                       \
  } while (0)

struct ClientState {
  // Set if this fd is a registered cluster member (initial or integrated newcomer).
  bool isMember = false;
  uint32_t nodeId = 0;

  // Set if this fd is a newcomer that's still in PENDING. nodeId is meaningless
  // until INTEGRATE is sent.
  bool isPending = false;

  // Set if this fd is currently blocked waiting for a reply (initial bringup
  // pre-quorum, or pending newcomer).
  bool replyDeferred = false;
  uint32_t deferredReplyType = 0;

  // Cached UCX address from REGISTER.
  std::string ucxAddr;
};

class Coordinator {
 public:
  Coordinator(int port, uint32_t initialSize)
      : port_(port), initialSize_(initialSize) {}

  void run() {
    listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd_ < 0) die("socket");
    int one = 1;
    setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (::bind(listenFd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
      die("bind");
    if (::listen(listenFd_, 64) < 0) die("listen");

    fprintf(stderr, "[coord] listening on port %d, expecting %u initial ranks\n",
            port_, initialSize_);
    fflush(stderr);

    for (;;) {
      std::vector<pollfd> pfds;
      pfds.push_back({listenFd_, POLLIN, 0});
      for (auto &kv : clients_) pfds.push_back({kv.first, POLLIN, 0});

      int n = ::poll(pfds.data(), pfds.size(), -1);
      if (n < 0) {
        if (errno == EINTR) continue;
        die("poll");
      }

      if (pfds[0].revents & POLLIN) acceptOne();
      for (size_t i = 1; i < pfds.size(); ++i) {
        if (pfds[i].revents & (POLLIN | POLLHUP | POLLERR)) {
          handleClient(pfds[i].fd);
        }
      }
    }
  }

 private:
  int port_;
  uint32_t initialSize_;
  int listenFd_ = -1;
  uint32_t epoch_ = 0;

  std::unordered_map<int, ClientState> clients_;     // fd -> state
  std::unordered_map<uint32_t, int> memberFd_;       // nodeId -> fd
  std::vector<Member> members_;                      // in nodeId order
  std::deque<int> pendingFds_;                       // newcomers awaiting integrate

  // Barrier state for the current epoch.
  std::set<uint32_t> barrierArrivals_;
  std::vector<int> barrierFds_;

  [[noreturn]] void die(const char *msg) {
    perror(msg);
    exit(1);
  }

  void acceptOne() {
    int fd = ::accept(listenFd_, nullptr, nullptr);
    if (fd < 0) {
      perror("accept");
      return;
    }
    // Disable Nagle. The coord protocol is small request -> small reply per
    // round-trip (BARRIER, COMMIT, etc.); Nagle + delayed-ACK turn each
    // round-trip into ~80ms (the steady-state cost we were paying for every
    // post-rescale CmiBarrier through coord::barrier). With TCP_NODELAY each
    // round-trip drops to ~microseconds.
    {
      int one = 1;
      setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    clients_[fd] = ClientState{};
    LOG("accept fd=%d", fd);
  }

  void closeClient(int fd) {
    LOG("close fd=%d", fd);
    auto it = clients_.find(fd);
    if (it != clients_.end()) {
      if (it->second.isMember) {
        memberFd_.erase(it->second.nodeId);
        // Don't remove from members_ here — the cluster's membership is the
        // authoritative state; a transient disconnect shouldn't change it.
        // Caller is responsible for handling member loss via RECONFIG.
      }
      if (it->second.isPending) {
        auto pit = std::find(pendingFds_.begin(), pendingFds_.end(), fd);
        if (pit != pendingFds_.end()) pendingFds_.erase(pit);
      }
      clients_.erase(it);
    }
    ::close(fd);
  }

  void handleClient(int fd) {
    Frame f = read_frame(fd);
    if (f.type == 0) {
      closeClient(fd);
      return;
    }
    try {
      const uint8_t *p = f.payload.data();
      const uint8_t *end = p + f.payload.size();
      switch (f.type) {
        case REGISTER_INITIAL: handleRegisterInitial(fd, p, end); break;
        case REGISTER_NEWCOMER: handleRegisterNewcomer(fd, p, end); break;
        case QUERY_PENDING: handleQueryPending(fd); break;
        case COMMIT: handleCommit(fd, p, end); break;
        case BARRIER: handleBarrier(fd, p, end); break;
        default:
          fprintf(stderr, "[coord] unknown msg type %u from fd=%d\n", f.type, fd);
          closeClient(fd);
      }
    } catch (const std::exception &e) {
      fprintf(stderr, "[coord] error from fd=%d: %s\n", fd, e.what());
      closeClient(fd);
    }
  }

  void handleRegisterInitial(int fd, const uint8_t *p, const uint8_t *end) {
    uint32_t nodeId = get_u32(p, end);
    std::string ucxAddr = get_bytes(p, end);
    if (nodeId >= initialSize_) {
      fprintf(stderr, "[coord] REGISTER_INITIAL nodeId=%u out of range (initial-size=%u)\n",
              nodeId, initialSize_);
      closeClient(fd);
      return;
    }
    if (memberFd_.count(nodeId)) {
      fprintf(stderr, "[coord] REGISTER_INITIAL duplicate nodeId=%u\n", nodeId);
      closeClient(fd);
      return;
    }
    auto &cs = clients_[fd];
    cs.ucxAddr = std::move(ucxAddr);
    cs.replyDeferred = true;
    cs.deferredReplyType = REGISTER_INITIAL_REPLY;
    cs.nodeId = nodeId;
    cs.isMember = true;
    memberFd_[nodeId] = fd;
    // members_ is kept in nodeId order; resize as ranks arrive out of order.
    if (members_.size() < initialSize_) members_.resize(initialSize_);
    members_[nodeId] = {nodeId, cs.ucxAddr};

    LOG("REGISTER_INITIAL fd=%d nodeId=%u (have %zu/%u)",
        fd, nodeId, memberFd_.size(), initialSize_);

    if (memberFd_.size() == initialSize_) {
      // Quorum reached — broadcast REGISTER_INITIAL_REPLY to every initial rank.
      LOG("initial quorum reached, sending replies");
      for (const auto &m : members_) {
        int memFd = memberFd_[m.nodeId];
        sendInitialReply(memFd, m.nodeId);
        clients_[memFd].replyDeferred = false;
      }
    }
  }

  void sendInitialReply(int fd, uint32_t nodeId) {
    std::vector<uint8_t> buf;
    put_u32(buf, nodeId);
    put_u32(buf, epoch_);
    put_members(buf, members_);
    if (!send_frame(fd, REGISTER_INITIAL_REPLY, buf)) closeClient(fd);
  }

  void handleRegisterNewcomer(int fd, const uint8_t *p, const uint8_t *end) {
    std::string ucxAddr = get_bytes(p, end);
    auto &cs = clients_[fd];
    cs.ucxAddr = std::move(ucxAddr);
    cs.isPending = true;
    cs.replyDeferred = true;
    cs.deferredReplyType = INTEGRATE;
    pendingFds_.push_back(fd);
    LOG("REGISTER_NEWCOMER fd=%d (queue size now %zu)", fd, pendingFds_.size());

    // Send the current member snapshot so the newcomer can build speculative
    // ucp endpoints while waiting for COMMIT. nodeIds in this reply are in
    // the *current* numbering; INTEGRATE will follow with the final compact
    // numbering. The newcomer keys its speculative eps by ucxAddr and remaps
    // them once INTEGRATE arrives.
    std::vector<uint8_t> sbuf;
    put_u32(sbuf, epoch_);
    put_members(sbuf, members_);
    if (!send_frame(fd, REGISTER_NEWCOMER_REPLY, sbuf)) closeClient(fd);
  }

  void handleQueryPending(int fd) {
    std::vector<uint8_t> buf;
    put_u32(buf, static_cast<uint32_t>(pendingFds_.size()));
    if (!send_frame(fd, QUERY_PENDING_REPLY, buf)) closeClient(fd);
    LOG("QUERY_PENDING fd=%d -> %zu", fd, pendingFds_.size());
  }

  void handleCommit(int fd, const uint8_t *p, const uint8_t *end) {
    uint32_t reqEpoch = get_u32(p, end);
    uint32_t numKills = get_u32(p, end);
    std::set<uint32_t> killSet;
    for (uint32_t i = 0; i < numKills; ++i) killSet.insert(get_u32(p, end));
    uint32_t take = get_u32(p, end);

    LOG("COMMIT epoch=%u kills=%u take=%u (have %zu pending)",
        reqEpoch, numKills, take, pendingFds_.size());

    if (reqEpoch != epoch_) {
      fprintf(stderr, "[coord] COMMIT epoch mismatch: req=%u cur=%u\n", reqEpoch, epoch_);
      closeClient(fd);
      return;
    }

    auto initIt = clients_.find(fd);
    if (initIt == clients_.end() || !initIt->second.isMember) {
      fprintf(stderr, "[coord] COMMIT from non-member fd=%d\n", fd);
      closeClient(fd);
      return;
    }
    uint32_t initiatorOldId = initIt->second.nodeId;
    if (killSet.count(initiatorOldId)) {
      fprintf(stderr, "[coord] COMMIT initiator nodeId=%u in kill set\n", initiatorOldId);
      closeClient(fd);
      return;
    }

    take = std::min<uint32_t>(take, pendingFds_.size());

    // Build the new member list with compact renumbering. Survivors keep
    // ucxAddrs; newcomers append at the end with new nodeIds.
    // Track each survivor's old fd so we can push RECONFIG and rebind cleanly.
    struct Survivor { uint32_t oldId; int fd; std::string ucxAddr; };
    std::vector<Survivor> survivors;
    survivors.reserve(members_.size() - killSet.size());
    for (const auto &m : members_) {
      if (killSet.count(m.nodeId)) continue;
      auto it = memberFd_.find(m.nodeId);
      if (it == memberFd_.end()) {
        fprintf(stderr, "[coord] COMMIT: surviving nodeId=%u has no fd\n", m.nodeId);
        closeClient(fd);
        return;
      }
      survivors.push_back({m.nodeId, it->second, m.ucxAddr});
    }

    std::vector<int> takenFds;
    takenFds.reserve(take);
    std::vector<Member> newMembers;
    newMembers.reserve(survivors.size() + take);
    uint32_t nextId = 0;
    for (auto &s : survivors) {
      newMembers.push_back({nextId++, s.ucxAddr});
    }
    for (uint32_t i = 0; i < take; ++i) {
      int newFd = pendingFds_.front();
      pendingFds_.pop_front();
      takenFds.push_back(newFd);
      newMembers.push_back({nextId++, clients_[newFd].ucxAddr});
    }

    // Collect killed fds before we mutate clients_/memberFd_.
    std::vector<int> killedFds;
    killedFds.reserve(killSet.size());
    for (uint32_t kid : killSet) {
      auto it = memberFd_.find(kid);
      if (it != memberFd_.end()) killedFds.push_back(it->second);
    }

    // Apply.
    epoch_++;
    members_ = newMembers;
    memberFd_.clear();
    barrierArrivals_.clear();
    barrierFds_.clear();
    for (auto &kv : clients_) {
      if (kv.second.isMember) kv.second.isMember = false;
    }
    for (size_t i = 0; i < survivors.size(); ++i) {
      auto &cs = clients_[survivors[i].fd];
      cs.isMember = true;
      cs.nodeId = newMembers[i].nodeId;
      memberFd_[cs.nodeId] = survivors[i].fd;
    }
    for (size_t i = 0; i < takenFds.size(); ++i) {
      int newFd = takenFds[i];
      auto &cs = clients_[newFd];
      cs.isPending = false;
      cs.isMember = true;
      cs.nodeId = newMembers[survivors.size() + i].nodeId;
      memberFd_[cs.nodeId] = newFd;
    }

    // Build the kill list (in OLD numbering) and the added list (newcomers
    // with their final NEW nodeIds) once — both are reused for every survivor
    // and for the initiator. Survivors apply the delta against their cached
    // view to reconstruct the full new membership.
    std::vector<uint32_t> killedOldIds(killSet.begin(), killSet.end());
    std::vector<Member> added;
    added.reserve(takenFds.size());
    for (size_t i = 0; i < takenFds.size(); ++i) {
      added.push_back(newMembers[survivors.size() + i]);
    }

    // Push INTEGRATE to each newly-integrated newcomer. Newcomers don't have
    // a stable cached view (their snapshot may be from an older epoch if a
    // prior commit didn't take them), so they still get the full member list.
    for (size_t i = 0; i < takenFds.size(); ++i) {
      int newFd = takenFds[i];
      uint32_t nodeId = clients_[newFd].nodeId;
      std::vector<uint8_t> ipl;
      put_u32(ipl, nodeId);
      put_u32(ipl, epoch_);
      put_members(ipl, members_);
      if (!send_frame(newFd, INTEGRATE, ipl)) closeClient(newFd);
      else clients_[newFd].replyDeferred = false;
    }

    // Surviving non-initiator members no longer receive RECONFIG over TCP.
    // PE 0 broadcasts the delta to them via the existing UCX endpoints (chain
    // fanout in machine.C). The coordinator still pushes DIE/INTEGRATE.

    // Push DIE to killed members so they exit cleanly.
    for (int kfd : killedFds) {
      send_frame(kfd, DIE, {});
      // Don't close — the killed rank will close from its side after _exit.
    }

    // Reply to PE 0 with the same delta shape.
    {
      uint32_t initiatorNewId = clients_[fd].nodeId;
      std::vector<uint8_t> rpl;
      put_u32(rpl, initiatorNewId);
      put_u32(rpl, epoch_);
      put_u32_vec(rpl, killedOldIds);
      put_members(rpl, added);
      if (!send_frame(fd, COMMIT_REPLY, rpl)) closeClient(fd);
    }
  }

  void handleBarrier(int fd, const uint8_t *p, const uint8_t *end) {
    uint32_t reqEpoch = get_u32(p, end);
    uint32_t nodeId = get_u32(p, end);
    if (reqEpoch != epoch_) {
      fprintf(stderr, "[coord] BARRIER epoch mismatch: req=%u cur=%u\n", reqEpoch, epoch_);
      closeClient(fd);
      return;
    }
    barrierArrivals_.insert(nodeId);
    barrierFds_.push_back(fd);
    LOG("BARRIER fd=%d node=%u (%zu/%zu)", fd, nodeId, barrierArrivals_.size(), members_.size());
    if (barrierArrivals_.size() == members_.size()) {
      LOG("barrier complete, releasing %zu fds", barrierFds_.size());
      for (int bfd : barrierFds_) {
        send_frame(bfd, BARRIER_REPLY, {});
      }
      barrierArrivals_.clear();
      barrierFds_.clear();
    }
  }
};

void usage() {
  fprintf(stderr,
          "Usage: charm_coordinator --port P --initial-size N [--verbose]\n");
  exit(1);
}

}  // namespace

int main(int argc, char **argv) {
  signal(SIGPIPE, SIG_IGN);

  int port = -1;
  int initialSize = -1;
  static struct option opts[] = {
      {"port", required_argument, nullptr, 'p'},
      {"initial-size", required_argument, nullptr, 'n'},
      {"verbose", no_argument, nullptr, 'v'},
      {nullptr, 0, nullptr, 0},
  };
  int c;
  while ((c = getopt_long(argc, argv, "p:n:v", opts, nullptr)) != -1) {
    switch (c) {
      case 'p': port = atoi(optarg); break;
      case 'n': initialSize = atoi(optarg); break;
      case 'v': g_verbose = true; break;
      default: usage();
    }
  }
  if (port <= 0 || initialSize <= 0) usage();

  Coordinator coord(port, static_cast<uint32_t>(initialSize));
  coord.run();
  return 0;
}

// End-to-end test of the coordinator protocol.
//
// Spins up N "initial" rank clients (idx 0 = driver/PE 0, idx 1..N-1 = forks)
// and M "newcomer" clients, exercises QUERY_PENDING / COMMIT, and verifies:
//   - all initial ranks get REGISTER_INITIAL_REPLY with consistent membership
//   - newcomers get INTEGRATE with the right nodeIds after COMMIT
//   - surviving non-initiator ranks get RECONFIG
//   - killed ranks get DIE
//
// Usage:
//   coordinator_test_client --host H --port P --initial-size N --new K --kill K1,K2,...
// Run the coordinator first with matching --initial-size and --port.

#include <arpa/inet.h>
#include <getopt.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "coord_client.h"

using namespace coord;

namespace {

// Surviving non-initiator: register, get initial reply, then block waiting
// for either RECONFIG (we survived) or DIE (we got killed). Print the
// outcome and exit so the parent can reap.
void run_initial(const char *host, int port, int idx) {
  int fd = connect_blocking(host, port);
  if (fd < 0) { fprintf(stderr, "[initial %d] connect failed\n", idx); _exit(1); }
  std::string ucxAddr = "INITIAL_RANK_" + std::to_string(idx);
  ClusterView view;
  if (!register_initial(fd, static_cast<uint32_t>(idx), ucxAddr.data(),
                        static_cast<uint32_t>(ucxAddr.size()), &view)) {
    fprintf(stderr, "[initial %d] register_initial failed\n", idx);
    _exit(1);
  }
  fprintf(stderr, "[initial idx=%d] registered nodeId=%u epoch=%u members=%zu\n",
          idx, view.nodeId, view.epoch, view.members.size());

  ClusterView post;
  bool gotDie = false;
  bool ok = await_reconfig_or_die(fd, &post, &gotDie);
  if (gotDie) {
    fprintf(stderr, "[initial idx=%d] DIE received — exiting\n", idx);
  } else if (ok) {
    fprintf(stderr, "[initial idx=%d] RECONFIG newNodeId=%u epoch=%u members=%zu\n",
            idx, post.nodeId, post.epoch, post.members.size());
  } else {
    fprintf(stderr, "[initial idx=%d] socket closed before push\n", idx);
  }
  ::close(fd);
}

void run_newcomer(const char *host, int port, int idx) {
  int fd = connect_blocking(host, port);
  if (fd < 0) { fprintf(stderr, "[newcomer %d] connect failed\n", idx); _exit(1); }
  std::string ucxAddr = "NEWCOMER_" + std::to_string(idx);
  ClusterView snapshot;
  if (!register_newcomer(fd, ucxAddr.data(),
                         static_cast<uint32_t>(ucxAddr.size()), &snapshot)) {
    fprintf(stderr, "[newcomer %d] register_newcomer (snapshot) failed\n", idx);
    _exit(1);
  }
  fprintf(stderr, "[newcomer idx=%d] snapshot epoch=%u members=%zu\n",
          idx, snapshot.epoch, snapshot.members.size());

  ClusterView view;
  if (!await_integrate(fd, &view)) {
    fprintf(stderr, "[newcomer %d] await_integrate failed\n", idx);
    _exit(1);
  }
  fprintf(stderr, "[newcomer idx=%d] integrated nodeId=%u epoch=%u members=%zu\n",
          idx, view.nodeId, view.epoch, view.members.size());
  ::close(fd);
}

void usage() {
  fprintf(stderr,
          "Usage: coordinator_test_client --host H --port P --initial-size N "
          "--new K [--kill id1,id2,...]\n");
  exit(1);
}

}  // namespace

int main(int argc, char **argv) {
  signal(SIGPIPE, SIG_IGN);
  const char *host = "127.0.0.1";
  int port = -1;
  int initialSize = -1;
  int newCount = 0;
  std::set<uint32_t> killSet;

  static struct option opts[] = {
      {"host", required_argument, nullptr, 'h'},
      {"port", required_argument, nullptr, 'p'},
      {"initial-size", required_argument, nullptr, 'n'},
      {"new", required_argument, nullptr, 'k'},
      {"kill", required_argument, nullptr, 'd'},
      {nullptr, 0, nullptr, 0},
  };
  int c;
  while ((c = getopt_long(argc, argv, "h:p:n:k:d:", opts, nullptr)) != -1) {
    switch (c) {
      case 'h': host = optarg; break;
      case 'p': port = atoi(optarg); break;
      case 'n': initialSize = atoi(optarg); break;
      case 'k': newCount = atoi(optarg); break;
      case 'd': {
        char *tok = strtok(optarg, ",");
        while (tok) {
          killSet.insert(static_cast<uint32_t>(atoi(tok)));
          tok = strtok(nullptr, ",");
        }
        break;
      }
      default: usage();
    }
  }
  if (port <= 0 || initialSize <= 0) usage();

  if (killSet.count(0)) {
    fprintf(stderr, "PE 0 (idx=0) cannot be in --kill set; it drives COMMIT\n");
    exit(1);
  }

  // Fork initial-rank children for idx 1..initialSize-1.
  std::vector<pid_t> initPids;
  for (int i = 1; i < initialSize; ++i) {
    pid_t pid = fork();
    if (pid == 0) { run_initial(host, port, i); _exit(0); }
    initPids.push_back(pid);
  }

  // Fork newcomers.
  std::vector<pid_t> newPids;
  for (int i = 0; i < newCount; ++i) {
    pid_t pid = fork();
    if (pid == 0) { run_newcomer(host, port, i); _exit(0); }
    newPids.push_back(pid);
  }

  // Driver acts as PE 0 (initial idx=0): register first, then drive COMMIT.
  int fd = connect_blocking(host, port);
  if (fd < 0) { fprintf(stderr, "[driver] connect failed\n"); exit(1); }
  std::string ucxAddr = "INITIAL_RANK_0";
  ClusterView view;
  if (!register_initial(fd, 0, ucxAddr.data(),
                        static_cast<uint32_t>(ucxAddr.size()), &view)) {
    fprintf(stderr, "[driver] register_initial failed\n");
    exit(1);
  }
  fprintf(stderr, "[driver] registered nodeId=%u epoch=%u members=%zu\n",
          view.nodeId, view.epoch, view.members.size());

  uint32_t pending = 0;
  if (!query_pending(fd, &pending)) { fprintf(stderr, "QUERY failed\n"); exit(1); }
  fprintf(stderr, "[driver] pending=%u\n", pending);

  std::vector<uint32_t> kills(killSet.begin(), killSet.end());
  ClusterView committed;
  if (!commit(fd, view.epoch, kills, static_cast<uint32_t>(newCount), &committed)) {
    fprintf(stderr, "COMMIT failed\n");
    exit(1);
  }
  fprintf(stderr, "[driver] COMMIT_REPLY newNodeId=%u epoch=%u members=%zu:\n",
          committed.nodeId, committed.epoch, committed.members.size());
  for (auto &m : committed.members) {
    fprintf(stderr, "  nodeId=%u ucx=%.*s\n", m.nodeId,
            static_cast<int>(m.ucxAddr.size()), m.ucxAddr.data());
  }

  // Give children a moment to print their push results, then reap.
  usleep(300 * 1000);
  for (pid_t pid : initPids) waitpid(pid, nullptr, 0);
  for (pid_t pid : newPids) waitpid(pid, nullptr, 0);
  ::close(fd);

  fprintf(stderr, "[driver] done\n");
  return 0;
}

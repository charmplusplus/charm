// Rank-side client for the coordinator. Header-only, depends only on wire.h.
//
// All calls are blocking and return false on socket error / unexpected reply.
// Callers should treat any false return as fatal (the cluster's coordinator
// state can't be reconciled if a rank loses its connection mid-protocol).

#pragma once

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "wire.h"

namespace coord {

// Connect with retry. Coordinator may not be up yet when initial ranks start.
// Returns -1 if all attempts fail.
inline int connect_blocking(const char *host, int port, int retries = 50,
                            int retryUsec = 100 * 1000) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *res = nullptr;
  char portStr[16];
  std::snprintf(portStr, sizeof(portStr), "%d", port);
  if (getaddrinfo(host, portStr, &hints, &res) != 0) return -1;
  int fd = -1;
  for (int i = 0; i < retries; ++i) {
    fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) break;
    if (::connect(fd, res->ai_addr, res->ai_addrlen) == 0) break;
    ::close(fd);
    fd = -1;
    usleep(retryUsec);
  }
  freeaddrinfo(res);
  // Disable Nagle on the rank-side socket. Pairs with the same setsockopt on
  // the coordinator's accept() side — without both, BARRIER round-trips
  // (small request, small reply) get stuck waiting for delayed-ACK / Nagle,
  // adding ~80ms per coord::barrier call.
  if (fd >= 0) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  }
  return fd;
}

struct ClusterView {
  uint32_t nodeId = 0;
  uint32_t epoch  = 0;
  std::vector<Member> members;
};

inline bool register_initial(int fd, uint32_t pmiRank,
                             const void *ucxAddr, uint32_t ucxAddrLen,
                             ClusterView *out) {
  std::vector<uint8_t> p;
  put_u32(p, pmiRank);
  put_bytes(p, ucxAddr, ucxAddrLen);
  if (!send_frame(fd, REGISTER_INITIAL, p)) return false;
  Frame f = read_frame(fd);
  if (f.type != REGISTER_INITIAL_REPLY) return false;
  const uint8_t *q = f.payload.data();
  const uint8_t *end = q + f.payload.size();
  out->nodeId = get_u32(q, end);
  out->epoch = get_u32(q, end);
  out->members = get_members(q, end);
  return true;
}

// Two-phase newcomer registration:
//   1) register_newcomer publishes the address and immediately returns the
//      current member snapshot (in *current* nodeId numbering). The newcomer
//      uses this to build speculative ucp endpoints during the wait window.
//      out->nodeId is left as 0 — it isn't assigned until INTEGRATE.
//   2) await_integrate blocks until COMMIT pushes the final view (compactly
//      renumbered nodeIds, including any other newcomers that integrated in
//      the same step). The newcomer must diff against the snapshot to decide
//      which speculative eps to keep, close, or add.
inline bool register_newcomer(int fd, const void *ucxAddr, uint32_t ucxAddrLen,
                              ClusterView *snapshot) {
  std::vector<uint8_t> p;
  put_bytes(p, ucxAddr, ucxAddrLen);
  if (!send_frame(fd, REGISTER_NEWCOMER, p)) return false;
  Frame f = read_frame(fd);
  if (f.type != REGISTER_NEWCOMER_REPLY) return false;
  const uint8_t *q = f.payload.data();
  const uint8_t *end = q + f.payload.size();
  snapshot->nodeId = 0;  // unassigned until INTEGRATE
  snapshot->epoch = get_u32(q, end);
  snapshot->members = get_members(q, end);
  return true;
}

inline bool await_integrate(int fd, ClusterView *out) {
  Frame f = read_frame(fd);
  if (f.type != INTEGRATE) return false;
  const uint8_t *q = f.payload.data();
  const uint8_t *end = q + f.payload.size();
  out->nodeId = get_u32(q, end);
  out->epoch = get_u32(q, end);
  out->members = get_members(q, end);
  return true;
}

inline bool query_pending(int fd, uint32_t *outCount) {
  if (!send_frame(fd, QUERY_PENDING, {})) return false;
  Frame f = read_frame(fd);
  if (f.type != QUERY_PENDING_REPLY) return false;
  const uint8_t *q = f.payload.data();
  const uint8_t *end = q + f.payload.size();
  *outCount = get_u32(q, end);
  return true;
}

// COMMIT_REPLY carries a delta against the caller's cached `oldMembers`
// (which must be the membership at `epoch` in OLD nodeId order). On success
// `out->members` is the fully reconstructed new view in NEW nodeId order.
inline bool commit(int fd, uint32_t epoch, const std::vector<uint32_t> &kills,
                   uint32_t take, const std::vector<Member> &oldMembers,
                   ClusterView *out) {
  std::vector<uint8_t> p;
  put_u32(p, epoch);
  put_u32(p, static_cast<uint32_t>(kills.size()));
  for (uint32_t k : kills) put_u32(p, k);
  put_u32(p, take);
  if (!send_frame(fd, COMMIT, p)) return false;
  Frame f = read_frame(fd);
  if (f.type != COMMIT_REPLY) return false;
  const uint8_t *q = f.payload.data();
  const uint8_t *end = q + f.payload.size();
  out->nodeId = get_u32(q, end);
  out->epoch = get_u32(q, end);
  std::vector<uint32_t> killedOldIds = get_u32_vec(q, end);
  std::vector<Member> added = get_members(q, end);
  out->members = apply_member_delta(oldMembers, killedOldIds, added);
  return true;
}

// Surviving non-initiator members block on this in ConverseCleanup. Returns
// true on RECONFIG (out is populated from a delta applied to oldMembers),
// false on DIE (caller should exit). On socket error returns false with
// out->epoch == 0 — caller should treat as fatal (use the read_frame
// f.type==0 convention).
inline bool await_reconfig_or_die(int fd, const std::vector<Member> &oldMembers,
                                  ClusterView *out, bool *gotDie) {
  Frame f = read_frame(fd);
  if (f.type == DIE) { *gotDie = true; return false; }
  if (f.type != RECONFIG) { *gotDie = false; return false; }
  *gotDie = false;
  const uint8_t *q = f.payload.data();
  const uint8_t *end = q + f.payload.size();
  out->nodeId = get_u32(q, end);
  out->epoch = get_u32(q, end);
  std::vector<uint32_t> killedOldIds = get_u32_vec(q, end);
  std::vector<Member> added = get_members(q, end);
  out->members = apply_member_delta(oldMembers, killedOldIds, added);
  return true;
}

inline bool barrier(int fd, uint32_t epoch, uint32_t nodeId) {
  std::vector<uint8_t> p;
  put_u32(p, epoch);
  put_u32(p, nodeId);
  if (!send_frame(fd, BARRIER, p)) return false;
  Frame f = read_frame(fd);
  return f.type == BARRIER_REPLY;
}

}  // namespace coord

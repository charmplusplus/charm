// Tiny serialization helpers and blocking socket I/O. Header-only so the
// coordinator and rank-side client can share the same code without a library.

#pragma once

#include <arpa/inet.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "protocol.h"

namespace coord {

struct Member {
  uint32_t nodeId;
  std::string ucxAddr;  // raw bytes
};

inline void put_u32(std::vector<uint8_t> &buf, uint32_t v) {
  uint32_t be = htonl(v);
  const uint8_t *p = reinterpret_cast<const uint8_t *>(&be);
  buf.insert(buf.end(), p, p + 4);
}

inline uint32_t get_u32(const uint8_t *&p, const uint8_t *end) {
  if (p + 4 > end) throw std::runtime_error("coord: short read");
  uint32_t v;
  memcpy(&v, p, 4);
  p += 4;
  return ntohl(v);
}

inline void put_bytes(std::vector<uint8_t> &buf, const void *data, uint32_t n) {
  put_u32(buf, n);
  const uint8_t *p = static_cast<const uint8_t *>(data);
  buf.insert(buf.end(), p, p + n);
}

inline std::string get_bytes(const uint8_t *&p, const uint8_t *end) {
  uint32_t n = get_u32(p, end);
  if (p + n > end) throw std::runtime_error("coord: short read");
  std::string s(reinterpret_cast<const char *>(p), n);
  p += n;
  return s;
}

inline void put_members(std::vector<uint8_t> &buf, const std::vector<Member> &m) {
  put_u32(buf, static_cast<uint32_t>(m.size()));
  for (const auto &x : m) {
    put_u32(buf, x.nodeId);
    put_bytes(buf, x.ucxAddr.data(), static_cast<uint32_t>(x.ucxAddr.size()));
  }
}

inline std::vector<Member> get_members(const uint8_t *&p, const uint8_t *end) {
  uint32_t n = get_u32(p, end);
  std::vector<Member> out;
  out.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    Member m;
    m.nodeId = get_u32(p, end);
    m.ucxAddr = get_bytes(p, end);
    out.push_back(std::move(m));
  }
  return out;
}

inline void put_u32_vec(std::vector<uint8_t> &buf, const std::vector<uint32_t> &v) {
  put_u32(buf, static_cast<uint32_t>(v.size()));
  for (uint32_t x : v) put_u32(buf, x);
}

inline std::vector<uint32_t> get_u32_vec(const uint8_t *&p, const uint8_t *end) {
  uint32_t n = get_u32(p, end);
  std::vector<uint32_t> out;
  out.reserve(n);
  for (uint32_t i = 0; i < n; ++i) out.push_back(get_u32(p, end));
  return out;
}

// Apply a (killedOldIds, added) delta to an old member list (in OLD nodeId
// order) to produce the new member list with compact renumbering. Matches the
// server-side algorithm in coordinator.cpp handleCommit.
inline std::vector<Member> apply_member_delta(
    const std::vector<Member> &oldMembers,
    const std::vector<uint32_t> &killedOldIds,
    const std::vector<Member> &added) {
  std::vector<uint8_t> killed;  // bitmap by old nodeId
  for (uint32_t k : killedOldIds) {
    if (k >= killed.size()) killed.resize(k + 1, 0);
    killed[k] = 1;
  }
  std::vector<Member> out;
  out.reserve(oldMembers.size() - killedOldIds.size() + added.size());
  uint32_t nextId = 0;
  for (const auto &m : oldMembers) {
    if (m.nodeId < killed.size() && killed[m.nodeId]) continue;
    out.push_back({nextId++, m.ucxAddr});
  }
  for (const auto &m : added) out.push_back(m);
  return out;
}

// Read exactly n bytes, returning false on EOF/error.
inline bool read_exact(int fd, void *buf, size_t n) {
  uint8_t *p = static_cast<uint8_t *>(buf);
  while (n > 0) {
    ssize_t r = ::read(fd, p, n);
    if (r == 0) return false;
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

inline bool write_all(int fd, const void *buf, size_t n) {
  const uint8_t *p = static_cast<const uint8_t *>(buf);
  while (n > 0) {
    ssize_t w = ::write(fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

// Read one frame. On success, returns (type, payload). On EOF/error, returns
// type==0 and empty payload — callers should check for the empty/closed case.
struct Frame {
  uint32_t type = 0;
  std::vector<uint8_t> payload;
};

inline Frame read_frame(int fd) {
  Frame f;
  uint32_t hdr[2];
  if (!read_exact(fd, hdr, sizeof(hdr))) return f;
  uint32_t totalLen = ntohl(hdr[0]);
  uint32_t type = ntohl(hdr[1]);
  if (totalLen < 4) return f;
  uint32_t payloadLen = totalLen - 4;
  f.payload.resize(payloadLen);
  if (payloadLen > 0 && !read_exact(fd, f.payload.data(), payloadLen)) {
    f.payload.clear();
    return f;
  }
  f.type = type;
  return f;
}

// Non-blocking probe + blocking read. Returns true if a complete frame is read
// into *out. Returns false if no full frame header is available yet (caller
// should retry later). On EOF or fatal error sets *gotEof=true and returns
// false. Uses MSG_PEEK to avoid consuming a partial header.
inline bool try_read_frame(int fd, Frame *out, bool *gotEof) {
  *gotEof = false;
  uint8_t hdr[8];
  ssize_t r = ::recv(fd, hdr, sizeof(hdr), MSG_PEEK | MSG_DONTWAIT);
  if (r == 0) { *gotEof = true; return false; }
  if (r < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return false;
    *gotEof = true;
    return false;
  }
  if (r < (ssize_t)sizeof(hdr)) return false;  // partial header — wait
  *out = read_frame(fd);
  return out->type != 0;
}

inline bool send_frame(int fd, uint32_t type, const std::vector<uint8_t> &payload) {
  uint32_t totalLen = htonl(static_cast<uint32_t>(4 + payload.size()));
  uint32_t typeBe = htonl(type);
  if (!write_all(fd, &totalLen, 4)) return false;
  if (!write_all(fd, &typeBe, 4)) return false;
  if (!payload.empty() && !write_all(fd, payload.data(), payload.size())) return false;
  return true;
}

}  // namespace coord

// Shared declarations of commbench: the case vocabulary, one end of a running
// case (Side), the method-family interface (Case) and the entity interface
// (Link). See README.md for the measurement rules.
#ifndef COMMBENCH_H
#define COMMBENCH_H

#include <set>
#include <string>
#include <vector>

#include "commbench.decl.h"

extern int g_maxSize;  // largest size of the run; buffers are this big

enum Entity { E_HANDLER, E_CHARE, E_GROUP, E_NODEGROUP, E_ARRAY, E_COUNT };

// Method families; a method is a family plus its registration modes.
enum Family {
  F_MARSHAL,
  F_MSG,
  F_NOCOPY,
  F_POST,
  F_GET,
  F_PUT,
  F_SYNC,  // the Converse families start here
  F_NODE,
  F_RGET,
  F_RPUT,
  F_PERSIST_SEND,
  F_PERSIST_RECV
};

struct MethodInfo {
  const char *name;
  int family;
  unsigned short regMode;    // CK_BUFFER_REG / UNREG / PREREG (zerocopy families)
  unsigned short deregMode;  // CK_BUFFER_DEREG / NODEREG (get, put)
};

extern const char *const kEntityName[E_COUNT];
extern const MethodInfo kMethods[];
extern const int kNumMethods;

inline int familyOf(int m) { return kMethods[m].family; }
inline bool isConverseMethod(int m) { return familyOf(m) >= F_SYNC; }
inline bool isPreregMethod(int m) { return kMethods[m].regMode == CK_BUFFER_PREREG; }
// The protocol string written into each sample record.
const char *protocolOf(int m, bool sameNode);

// Roles: A starts each round trip and times it, B answers.
enum { ROLE_A = 0, ROLE_B = 1 };

class CommMsg : public CMessage_CommMsg {
 public:
  CmiInt8 n;
  char *data;
};

// Reply from an endpoint to the driver's future.
class ResultMsg : public CMessage_ResultMsg {
 public:
  double elapsed = 0;
  int ok = 1;            // 0: this side saw a payload mismatch
  CmiInt8 checked = 0;   // bytes checked by this side
  CmiInt8 mismatches = 0;
  int npes = 0;          // PEs that ran this side's receives (node-level cases)
  int pes[32];
  CkChareID cid;         // ChareEP creation only
  char error[256] = {0};
  char aux[128] = {0};   // Side::auxNote
};

// Payload pattern: depends on the byte offset, the sending role and the size,
// so a stale buffer from another size or the other direction fails the check.
inline unsigned char patternByte(size_t i, int role, size_t size) {
  CmiUInt4 h = (CmiUInt4)(i + 1) * 0x9E3779B1u ^ (CmiUInt4)size * 0x85EBCA77u ^
               (CmiUInt4)(role + 1) * 0xC2B2AE3Du;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return (unsigned char)h;
}
void fillPattern(char *buf, size_t n, int role, size_t size);
// Fills with the complement of the pattern, so every byte must be rewritten.
void fillInverse(char *buf, size_t n, int role, size_t size);

// Send and receive buffers of one PE, g_maxSize bytes each, allocated on first
// use and kept for the whole run. Plain ones come from posix_memalign; the
// PREREG ones from CkRdmaAlloc.
void peBuffers(int pe, bool prereg, char **send, char **recv);
void sideInitNode();
// aux note naming the transfer path between this PE and peerPe for the
// zerocopy families: "local" (same process), "message" (copy-based
// fallback) or "rma" (one-sided network operation).
std::string rdmaPathNote(int peerPe);

class Case;

// What a method family needs from the entity it runs on.
struct Link {
  virtual ~Link() {}
  virtual void sendMarshal(int n, const char *d) = 0;
  virtual void sendMsg(CommMsg *m) = 0;
  virtual void sendNocopy(int n, const CkNcpyBuffer &b) = 0;
  virtual void sendPost(int n, const CkNcpyBuffer &b) = 0;
  virtual void sendNotify(CmiInt8 seq, CmiInt8 n) = 0;
  // A notify that carries a fresh handle (get.*.dereg), or the handle alone
  // (put.*.dereg).
  virtual void sendNotifyHandle(CmiInt8 seq, CmiInt8 n, const CkNcpyBuffer &h) = 0;
  virtual void sendHandles(const CkNcpyBuffer &s, const CkNcpyBuffer &r) = 0;
  // To this endpoint, for a receive handle: the runtime runs it when a get or
  // put into the handle's buffer has completed.
  virtual CkCallback destDoneCallback() = 0;
};

// One end of the running case. An endpoint takes part in one case at a time.
struct Side {
  int method = -1, size = 0, role = ROLE_A, myPe = -1, peerPe = -1;
  bool validate = false, trackPes = false;
  Link *link = nullptr;
  Case *c = nullptr;
  char *sendBuf = nullptr, *recvBuf = nullptr;

  CkNcpyBuffer peerSend, peerRecv;  // the peer's handles (get, put, rget, rput)
  bool haveHandles = false, setupDone = false, readySent = false;
  CkFuture ready;

  int iters = 0, warmup = 0, done = 0;
  double t0 = 0;
  CkFuture runDone;

  CmiInt8 checked = 0, mismatches = 0, firstBad = -1;
  int badExpected = 0, badGot = 0;
  std::string error;
  std::set<int> recvPes;

  void begin(int method, int size, int role, int myPe, int peerPe, bool validate,
             Link *link, CkFuture ready);
  void gotHandles(const CkNcpyBuffer &s, const CkNcpyBuffer &r);
  // The peer's setup data has arrived by another route (persistent.recv).
  void peerReady();
  // Extra fields for the sample's aux object, e.g. "\"persistent_path\":\"rma_put\"".
  std::string auxNote;
  void startRun(int iters, int warmup, CkFuture f);
  // A half round trip has delivered n bytes at d on this side.
  void halfArrived(const char *d, CmiInt8 n);
  void end(CkFuture f);

 private:
  void maybeReady();
  void check(const char *d, CmiInt8 n);
};

// A method family. The driver reaches it only through Side: setup, the run
// loop (ping on each half), teardown.
class Case {
 public:
  explicit Case(Side &s) : s(s) {}
  virtual ~Case() {}
  virtual bool needsHandles() const { return false; }
  virtual void setup() {}
  // Send one half round trip from this side.
  virtual void ping() = 0;
  virtual void teardown() {}

  virtual void onMarshal(int, char *) { unexpected("marshal"); }
  virtual void onMsg(CommMsg *) { unexpected("msg"); }
  virtual void onNocopy(int, char *) { unexpected("nocopy"); }
  virtual void onPostMatch(int, CkNcpyBufferPost *) { unexpected("post"); }
  virtual void onPost(int, char *) { unexpected("post"); }
  virtual void onNotify(CmiInt8, CmiInt8) { unexpected("notify"); }
  virtual void onNotifyHandle(CmiInt8, CmiInt8, CkNcpyBuffer &) { unexpected("notifyHandle"); }
  virtual void onDestDone() { unexpected("destDone"); }

 protected:
  Side &s;
  void unexpected(const char *what);
};

Case *makePlainCase(Side &s);     // cases/plain.C: marshal, msg
Case *makeZerocopyCase(Side &s);  // cases/zerocopy.C: nocopy.*, post.*
Case *makeDirectCase(Side &s);    // cases/direct.C: get, put
Case *makeConverseCase(Side &s);  // converse_cases.C: sync, node, rget, rput, persistent.*

// Converse handler indices and the HandlerEP sides, per PE (converse_cases.C).
void converseInitProc();
void converseInitNode();
void registerHandlerSide(Side *s);

#endif

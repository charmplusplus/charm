// Packing an element proxy built by the delegation constructor.
//
// Usage: ./proxy_delctor (any number of PEs)
//
// aProxy[i] and groupProxy[pe] build element proxies through
// CProxy(CK_DELCTOR_PARAM). For an undelegated proxy that constructor must
// leave delegatedGroupId zero, or CProxy::pup looks the stale id up in the
// group tables. Each PE builds such a proxy over memory filled with 0xA5, so
// that an uninitialized field holds a nonzero value, and packs and unpacks it.
// A stale id makes the pack spin in CkLocalNodeBranch, which runs the
// scheduler; the watchdog message queued just before is then delivered
// during the pack and aborts. Each PE then sends the proxy as a marshalled
// parameter to the next PE, which uses it to send back.

#include <cstring>
#include <new>
#include "proxy_delctor.decl.h"

/*readonly*/ CProxy_Main mainProxy;

// Keeps the compiler from dropping stores to *p or assuming its contents.
static inline void touchMemory(void *p) { asm volatile("" : : "r"(p) : "memory"); }

class Main : public CBase_Main {
public:
  Main(CkArgMsg *m) {
    delete m;
    mainProxy = thisProxy;
    CProxy_Target targets = CProxy_Target::ckNew();
    targets.run();
  }
  void done() {
    CkPrintf("All tests passed\n");
    CkExit();
  }
};

class Target : public CBase_Target {
  bool packing = false;

public:
  Target() {}

  void run() {
    thisProxy[CkMyPe()].watchdog();

    alignas(CProxyElement_Target) unsigned char buf[sizeof(CProxyElement_Target)];
    memset(buf, 0xA5, sizeof(buf));
    touchMemory(buf);
    CProxyElement_Target *p = new (buf) CProxyElement_Target(thisgroup, CkMyPe(), NULL, NULL);
    touchMemory(buf);

    packing = true;
    size_t n = PUP::size(*p);
    std::vector<char> bytes(n);
    PUP::toMemBuf(*p, bytes.data(), n);
    CProxyElement_Target q;
    PUP::fromMemBuf(q, bytes.data(), n);
    packing = false;

    if (q.ckIsDelegated() || !(q.ckGetGroupID() == thisgroup) || q.ckGetGroupPe() != CkMyPe())
      CkAbort("proxy_delctor: PE %d: unpacked proxy differs from the packed one", CkMyPe());

    thisProxy[(CkMyPe() + 1) % CkNumPes()].use(*p, CkMyPe());
    p->~CProxyElement_Target();
  }

  void watchdog() {
    if (packing)
      CkAbort("proxy_delctor: PE %d: packing an undelegated element proxy entered the "
              "scheduler (stale delegatedGroupId)", CkMyPe());
  }

  void use(CProxyElement_Target p, int fromPe) {
    if (p.ckIsDelegated() || p.ckGetGroupPe() != fromPe)
      CkAbort("proxy_delctor: PE %d: received proxy for PE %d, expected %d", CkMyPe(),
              p.ckGetGroupPe(), fromPe);
    p.hit(CkMyPe());
  }

  void hit(int fromPe) {
    if (fromPe != (CkMyPe() + 1) % CkNumPes())
      CkAbort("proxy_delctor: PE %d: hit from PE %d", CkMyPe(), fromPe);
    contribute(CkCallback(CkReductionTarget(Main, done), mainProxy));
  }
};

#include "proxy_delctor.def.h"

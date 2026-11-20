// nocopy.{reg,unreg,prereg}: entry recvNocopy(int n, nocopy char d[n]), sent
// with CkSendBuffer(buf, mode) on every send.
// post.{reg,unreg,prereg}: entry recvPost(int n, nocopypost char d[n]); the
// post entry sets the receive mode, then calls CkMatchBuffer and CkPostBuffer
// with this side's receive buffer.
// PREREG modes use buffers from CkRdmaAlloc on both sides.
#include "../commbench.h"

namespace {

class NocopyCase : public Case {
 public:
  using Case::Case;
  void setup() override { s.auxNote = rdmaPathNote(s.peerPe); }
  void ping() override {
    s.link->sendNocopy(s.size, CkSendBuffer(s.sendBuf, kMethods[s.method].regMode));
  }
  void onNocopy(int n, char *d) override { s.halfArrived(d, n); }
};

class PostCase : public Case {
  // Unique per PE while the transfer is in flight; one is in flight at a time.
  int tag() const { return 0x6300 + s.role; }

 public:
  using Case::Case;
  void setup() override { s.auxNote = rdmaPathNote(s.peerPe); }
  void ping() override {
    s.link->sendPost(s.size, CkSendBuffer(s.sendBuf, kMethods[s.method].regMode));
  }
  void onPostMatch(int n, CkNcpyBufferPost *post) override {
    if (n != s.size) CkAbort("commbench post: %d bytes announced, %d expected\n", n, s.size);
    post[0].regMode = kMethods[s.method].regMode;
    CkMatchBuffer(post, 0, tag());
    CkPostBuffer(s.recvBuf, (size_t)n, tag());
  }
  void onPost(int n, char *d) override {
    if (d != s.recvBuf) CkAbort("commbench post: data did not land in the posted buffer\n");
    s.halfArrived(d, n);
  }
};

}  // namespace

Case *makeZerocopyCase(Side &s) {
  if (familyOf(s.method) == F_NOCOPY) return new NocopyCase(s);
  return new PostCase(s);
}

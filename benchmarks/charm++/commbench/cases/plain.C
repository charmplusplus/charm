// marshal: entry recvMarshal(int n, char d[n]); the runtime packs the payload
// from sendBuf into the message, and the receiver finds it inside.
// msg: one varsize CommMsg circulates; each side sends back the message it
// received, so the timed loop allocates nothing. Under --source user each
// send first copies sendBuf into the message.
#include "../commbench.h"

namespace {

class MarshalCase : public Case {
 public:
  using Case::Case;
  void setup() override {
    // The message the generated recvMarshal sender builds: n, the array's
    // offset and count, aligned to 16 B, then the array.
    CkMarshallMsg *probe = CkAllocateMarshallMsg(16 + s.size);
    s.msgBytes = UsrToEnv(probe)->getTotalsize();
    delete probe;
  }
  void ping() override { s.link->sendMarshal(s.size, s.sendBuf); }
  void onMarshal(int n, char *d) override { s.halfArrived(d, n); }
};

class MsgCase : public Case {
  CommMsg *held = nullptr;  // the message, while it is on this side

 public:
  using Case::Case;
  void setup() override {
    CommMsg *probe = new (s.size) CommMsg;
    s.msgBytes = UsrToEnv(probe)->getTotalsize();
    delete probe;
    if (s.role == ROLE_A) {
      held = new (s.size) CommMsg;
      held->n = s.size;
    }
  }
  void ping() override {
    if (!held) CkAbort("commbench msg: ping without the message\n");
    s.fillOutgoing(held->data);
    CommMsg *m = held;
    held = nullptr;
    s.link->sendMsg(m);
  }
  void onMsg(CommMsg *m) override {
    held = m;
    s.halfArrived(m->data, m->n);
  }
  void teardown() override {
    delete held;
    held = nullptr;
  }
};

}  // namespace

Case *makePlainCase(Side &s) {
  if (familyOf(s.method) == F_MARSHAL) return new MarshalCase(s);
  return new MsgCase(s);
}

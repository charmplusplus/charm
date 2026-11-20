// marshal: entry recvMarshal(int n, char d[n]), the payload copied in and out.
// msg: one varsize CommMsg circulates; each side sends back the message it
// received, so the timed loop allocates nothing.
#include "../commbench.h"

namespace {

class MarshalCase : public Case {
 public:
  using Case::Case;
  void ping() override { s.link->sendMarshal(s.size, s.sendBuf); }
  void onMarshal(int n, char *d) override { s.halfArrived(d, n); }
};

class MsgCase : public Case {
  CommMsg *held = nullptr;  // the message, while it is on this side

 public:
  using Case::Case;
  void setup() override {
    if (s.role == ROLE_A) {
      held = new (s.size) CommMsg;
      held->n = s.size;
    }
  }
  void ping() override {
    if (!held) CkAbort("commbench msg: ping without the message\n");
    if (s.validate) fillPattern(held->data, s.size, s.role, s.size);
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

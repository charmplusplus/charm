#include "sdag.h"

namespace SDAG {
  PUPable_def(MsgClosure);
  PUPable_def(ForallClosure);
  PUPable_def(TransportableBigSimLog);
  PUPable_def(CCounter);
  PUPable_def(CSpeculator);
  PUPable_def(Buffer);
  PUPable_def(Continuation);

  void registerPUPables() {
    PUPable_reg(MsgClosure);
    PUPable_reg(ForallClosure);
    PUPable_reg(TransportableBigSimLog);
    PUPable_reg(CCounter);
    PUPable_reg(CSpeculator);
    PUPable_reg(Buffer);
    PUPable_reg(Continuation);
  }
}

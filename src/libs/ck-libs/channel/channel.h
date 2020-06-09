#ifndef _CHANNEL_H_
#define _CHANNEL_H_

#include "channel.decl.h"
#include <queue>

template<typename T>
class CkChannel : public CBase_CkChannel<T> {
public:
  CkChannel() {}
  CkChannel(CkMigrateMessage*) {}

  void send(T value) {
    data.push(value);
    if (!waiting.empty()) {
      auto front = waiting.front();
      waiting.pop();
      CthAwaken(front);
    }
  }

  inline T waitForData() {
    if (data.empty()) {
      waiting.push(CthSelf());
      CthSuspend();
    }
    auto front = data.front();
    data.pop();
    return front;
  }

  T receive() {
    return waitForData();
  }

  void receive(CkCallback cb) {
    auto value = waitForData();
    PUP::sizer szr;
    szr | value;
    CkMarshallMsg *msg = CkAllocateMarshallMsg(szr.size(), NULL);
    PUP::toMem ppr((void *)msg->msgBuf);
    ppr | value;
    cb.send(msg);
  }
private:
  std::queue<CthThread> waiting;
  std::queue<T> data;
};

#define CK_TEMPLATES_ONLY
#include "channel.def.h"
#undef CK_TEMPLATES_ONLY

#endif

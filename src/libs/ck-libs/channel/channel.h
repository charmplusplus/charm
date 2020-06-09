#ifndef _CHANNEL_H_
#define _CHANNEL_H_

#include "channel.decl.h"

#include <array>
#include <queue>
#include <utility>
#include <memory>

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

  T receive() {
    if (data.empty()) {
      waiting.push(CthSelf());
      CthSuspend();
    }
    auto front = data.front();
    data.pop();
    return front;
  }

  void receive(CkCallback cb) {
    auto value = this->receive();
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

template<int N>
class CkMultiWait {
  struct CkWaitRequest {
    CthThread thread;
    bool expired;
    CkWaitRequest(const CthThread& thread_)
    : thread(thread_), expired(false) { }
  };
  std::array<std::queue<std::shared_ptr<CkWaitRequest>>, N> waiting;
public:
  // Returns true when a thread is waiting on a channel
  // Stores the waiting thread in the reference argument
  bool hasWaiter(int channel, CthThread &thread) {
    if (waiting[channel].empty()) {
      return false;
    } else {
      // Pop a request from the wait-queue
      auto request = waiting[channel].front();
      waiting[channel].pop();
      // If it has expired
      if (request->expired) {
        // Recurse and check for another waiter
        return hasWaiter(channel, thread);
      } else {
        // Otherwise, mark it as expired
        request->expired = true;
        // And set the return values
        thread = request->thread;
        return true;
      }
    }
  }
  // Places a waiter into the queue for the specified channel
  void wait(int channel, const CthThread &thread) {
    waiting[channel].emplace(new CkWaitRequest(thread));
  }
  // Places a (common) waiter into the queue for all specified channels
  void waitForAny(int* channels, int numChannels, const CthThread &thread) {
    std::shared_ptr<CkWaitRequest> ptr(new CkWaitRequest(thread));
    for (int i = 0; i < numChannels; i++) {
      waiting[channels[i]].push(ptr);
    }
  }
};

template<int N, typename T>
class CkMultiChannel : public CBase_CkMultiChannel<N, T> {
  template<typename Value>
  inline void sendToCallback(CkCallback cb, Value value) {
    PUP::sizer szr;
    szr | value;
    CkMarshallMsg *msg = CkAllocateMarshallMsg(szr.size(), NULL);
    PUP::toMem ppr((void *)msg->msgBuf);
    ppr | value;
    cb.send(msg);
  }
public:
  CkMultiChannel() {}
  CkMultiChannel(CkMigrateMessage*) {}

  void send(int channel, T value) {
    channel = channel % N;
    data[channel].push(value);
    CthThread thread;
    if (waiting.hasWaiter(channel, thread)) {
      CthAwaken(thread);
    }
  }

  T receive(int channel) {
    channel = channel % N;
    if (data[channel].empty()) {
      waiting.wait(channel, CthSelf());
      CthSuspend();
    }
    auto front = data[channel].front();
    data[channel].pop();
    return front;
  }

  void receive(int channel, CkCallback cb) {
    sendToCallback(cb, this->receive(channel));
  }

  void waitAll(int* channels, int numChannels) {
    for (int i = 0; i < numChannels; i++) {
      this->receive(channels[i]);
    }
  }

  void waitN(int channel, int numMsgs) {
    for (int i = 0; i < numMsgs; i++) {
      this->receive(channel);
    }
  }

  int checkChannels(int* channels, int numChannels) {
    for (int i = 0; i < numChannels; i++) {
      int channel = channels[i] % N;
      if (!data[channel].empty()) {
        return channel;
      }
    }
    return -1;
  }

  std::pair<int, T> receiveFromAny(int* channels, int numChannels) {
    int which = checkChannels(channels, numChannels);
    if (which < 0) {
      waiting.waitForAny(channels, numChannels, CthSelf());
      CthSuspend();
      which = checkChannels(channels, numChannels);
    }
    auto front = data[which].front();
    data[which].pop();
    return std::make_pair(which, front);
  }

  void receiveFromAny(int* channels, int numChannels, CkCallback cb) {
    sendToCallback(cb, this->receiveFromAny(channels, numChannels));
  }
private:
  CkMultiWait<N> waiting;
  std::array<std::queue<T>, N> data;
};

#define CK_TEMPLATES_ONLY
#include "channel.def.h"
#undef CK_TEMPLATES_ONLY

#endif

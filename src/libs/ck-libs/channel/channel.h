#ifndef _CHANNEL_H_
#define _CHANNEL_H_

#include "channel.decl.h"

#include <array>
#include <queue>
#include <utility>
#include <memory>

namespace ck {
  namespace channel {
    template<typename Value>
    inline void sendToCallback(CkCallback cb, Value value) {
      PUP::sizer szr;
      szr | value;
      CkMarshallMsg *msg = CkAllocateMarshallMsg(szr.size(), NULL);
      PUP::toMem ppr((void *)msg->msgBuf);
      ppr | value;
      cb.send(msg);
    }

    struct WaitRequest {
      CthThread thread;
      bool expired;
      WaitRequest(const CthThread& thread_)
      : thread(thread_), expired(false) { }
    };

    template<int N>
    class MultiWait {
      std::array<std::queue<std::shared_ptr<WaitRequest>>, N> waiting;
    public:
      // Returns true when a thread is waiting on a channel
      // Stores the waiting thread in the reference argument
      bool hasWaiter(int channel, CthThread &thread) {
        // While a thread is waiting on a channel
        while (!waiting[channel].empty()) {
          // Pop a request from the wait-queue
          auto request = waiting[channel].front();
          waiting[channel].pop();
          // If it is not expired
          if (!request->expired) {
            // Mark it as expired
            request->expired = true;
            // Return thread
            thread = request->thread;
            return true;
          }
        }
        // Return nothing
        return false;
      }
      // Places a waiter into the queue for the specified channel
      void wait(int channel, const CthThread &thread) {
        waiting[channel].emplace(new WaitRequest(thread));
      }
      // Places a (common) waiter into the queue for all specified channels
      void waitForAny(int* channels, int numChannels, const CthThread &thread) {
        std::shared_ptr<WaitRequest> ptr(new WaitRequest(thread));
        for (int i = 0; i < numChannels; i++) {
          waiting[channels[i]].push(ptr);
        }
      }
    };
  }
}

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

  void receiveAsync(CkCallback cb) {
    ck::channel::sendToCallback(cb, this->receive());
  }

  std::vector<T> receiveN(int numMsgs) {
    std::vector<T> res;
    for (int i = 0; i < numMsgs; i++) {
      res.push_back(this->receive());
    }
    return res;
  }

  void receiveNAsync(int numMsgs, CkCallback cb) {
    ck::channel::sendToCallback(cb, this->receiveN(numMsgs));
  }

  void waitN(int numMsgs) {
    for (int i = 0; i < numMsgs; i++) {
      this->receive();
    }
  }
private:
  std::queue<CthThread> waiting;
  std::queue<T> data;
};

template<int N, typename T>
class CkMultiChannel : public CBase_CkMultiChannel<N, T> {
  int checkChannels(int* channels, int numChannels) {
    for (int i = 0; i < numChannels; i++) {
      int channel = channels[i] % N;
      if (!data[channel].empty()) {
        return channel;
      }
    }
    return -1;
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

  void receiveAsync(int channel, CkCallback cb) {
    ck::channel::sendToCallback(cb, this->receive(channel));
  }

  std::vector<T> receiveAll(int* channels, int numChannels) {
    std::vector<T> res;
    for (int i = 0; i < numChannels; i++) {
      res.push_back(this->receive(channels[i]));
    }
    return res;
  }

  void receiveAllAsync(int* channels, int numChannels, CkCallback cb) {
    ck::channel::sendToCallback(cb, this->receiveAll(channels, numChannels));
  }

  void waitAll(int* channels, int numChannels) {
    for (int i = 0; i < numChannels; i++) {
      this->receive(channels[i]);
    }
  }

  int waitAny(int* channels, int numChannels) {
    int which = checkChannels(channels, numChannels);
    if (which < 0) {
      waiting.waitForAny(channels, numChannels, CthSelf());
      CthSuspend();
      which = checkChannels(channels, numChannels);
    }
    return which;
  }

std::vector<T> receiveN(int channel, int numMsgs) {
    std::vector<T> res;
    for (int i = 0; i < numMsgs; i++) {
      res.push_back(this->receive(channel));
    }
    return res;
  }

  void receiveNAsync(int channel, int numMsgs, CkCallback cb) {
    ck::channel::sendToCallback(cb, this->receiveN(channel, numMsgs));
  }

  void waitN(int channel, int numMsgs) {
    for (int i = 0; i < numMsgs; i++) {
      this->receive(channel);
    }
  }

  std::pair<int, T> receiveAny(int* channels, int numChannels) {
    int which = this->waitAny(channels, numChannels);
    auto front = data[which].front();
    data[which].pop();
    return std::make_pair(which, front);
  }

  void receiveAnyAsync(int* channels, int numChannels, CkCallback cb) {
    ck::channel::sendToCallback(cb, this->receiveAny(channels, numChannels));
  }
private:
  ck::channel::MultiWait<N> waiting;
  std::array<std::queue<T>, N> data;
};

#define CK_TEMPLATES_ONLY
#include "channel.def.h"
#undef CK_TEMPLATES_ONLY

#endif

#ifndef __CK_FUTURE_H__
#define __CK_FUTURE_H__

#include "charm++.h"
#include <type_traits>

namespace ck {
  template<typename T>
  class future {
    CkFuture handle_;
public:
    future() {
      handle_ = CkCreateFuture();
    }

    future(const future<T>& other) {
      handle_ = other.handle_;
    }

    T get() {
      CkMarshallMsg *msg = (CkMarshallMsg *)CkWaitFuture(handle_);
      PUP::fromMem p(msg->msgBuf);
      PUP::detail::TemporaryObjectHolder<T> holder;
      p | holder;
      delete msg;
      return std::move(holder.t);
    }

    void set(const T& value) {
      PUP::sizer s;
      s | (typename std::decay<decltype(value)>::type&) value;
      CkMarshallMsg *msg = CkAllocateMarshallMsg(s.size(), nullptr);
      PUP::toMem p((void *)msg->msgBuf);
      p | (typename std::decay<decltype(value)>::type&) value;
      CkSendToFuture(handle_, msg);
    }

    bool probe() {
      return CkProbeFuture(handle_);
    }

    void release() {
      CkReleaseFuture(handle_);
    }

    void pup(PUP::er& p){
      p | handle_;
    }
  };
}

#endif

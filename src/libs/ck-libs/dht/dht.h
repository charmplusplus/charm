#ifndef __CK_DHT_H__
#define __CK_DHT_H__

#include <CkDhtManager.h>

namespace ck {
template <typename K, typename V> class dht {
  CProxy_CkDhtManager<size_t, V> proxy_;
  std::hash<K> hash_fn_;
public:
  dht(const dht<K, V> &other) { proxy_ = other.proxy_; }

  dht() { proxy_ = CProxy_CkDhtManager<size_t, V>::ckNew(); }

  inline void request(K key, ck::future<V> f) {
    auto key_ = hash_fn_(key);
    proxy_[key_ % CkNumNodes()].request(key_, f);
  }

  inline ck::future<V> request(K key) {
    ck::future<V> f;
    request(key, f);
    return std::move(f);
  }

  void insert(K key, V value) {
    auto key_ = hash_fn_(key);
    proxy_[key_ % CkNumNodes()].insert(key_, value);
  }

  void erase(K key) {
    auto key_ = hash_fn_(key);
    proxy_[key_ % CkNumNodes()].remove(key_);
  }

  void pup(PUP::er &p) { p | proxy_; }
};
}

#endif
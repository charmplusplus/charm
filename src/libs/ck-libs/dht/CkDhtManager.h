#ifndef __CKDHTMANAGER_H__
#define __CKDHTMANAGER_H__

#include "CkDhtManager.decl.h"
#include <map>
#include <vector>

template <typename CkDhtKey, typename CkDhtValue>
class CkDhtManager : public CBase_CkDhtManager<CkDhtKey, CkDhtValue> {
  using CkDhtFuture = ck::future<CkDhtValue>;
  std::map<CkDhtKey, CkDhtValue> map_;
  std::map<CkDhtKey, std::vector<CkDhtFuture>> pending_;
  CkDhtManager_SDAG_CODE

public:
  void request(CkDhtKey key, CkDhtFuture f) {
    CkAssert((key % CkNumNodes()) == CkMyNode());
    if (map_.find(key) == map_.end()) {
      if (pending_.find(key) == pending_.end()) {
        pending_[key] = { f };
      } else {
        pending_[key].push_back(f);
      }
    } else {
      f.set(map_[key]);
    }
  }

  void insert(CkDhtKey key, CkDhtValue value) {
    CkAssert((key % CkNumNodes()) == CkMyNode());
    map_[key] = value;
    if (pending_.find(key) != pending_.end()) {
      for (auto f : pending_[key]) {
        f.set(value);
      }
      pending_.erase(key);
    }
  }

  void remove(CkDhtKey key) {
    CkAssert((key % CkNumNodes()) == CkMyNode());
    CkAssert(map_.find(key) != map_.end());
    map_.erase(key);
  }
};

#define CK_TEMPLATES_ONLY
#include "CkDhtManager.def.h"
#undef CK_TEMPLATES_ONLY

#endif

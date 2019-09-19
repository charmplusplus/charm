#include <vector>
#include <unordered_map>
#include <pup_stl.h>

class IDM
{
public:

  ~IDM() {}

  void pup(PUP::er &p) {
    p|data;
  }

  size_t numDests() {
    return data.size();
  }

  // pe -> vector of (obj_local_id, dest)
  // this map stores, for a pe, the list of objects that need to migrate to a destination PE
  // in a different subtree
  std::unordered_map<int, std::vector<std::pair<int,int>>> data;
};

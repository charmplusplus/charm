#include <unordered_map>
#include <vector>

// pe -> vector of (obj_local_id, dest)
// this map stores, for a pe, the list of objects that need to migrate to a
// destination PE in a different subtree

using IDM = std::unordered_map<int, std::vector<std::pair<int, int>>>;

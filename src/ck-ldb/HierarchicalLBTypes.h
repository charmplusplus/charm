/**
 * Author: Jonathan Lifflander
 *
 * Hierarchical Greedy LB based on the description in this paper:
 *
 * HPDC 12:
 *   "Work Stealing and Persistence-based Load Balancers for Iterative
 *    Overdecomposed Applications"
 *    Jonathan Lifflander, Sriram Krishnamoorthy, Laxmikant V. Kale
 *
*/

#if ! defined _HIERLB_TYPES_H_
#define _HIERLB_TYPES_H_

#include <list>
#include <cstdint>
#include <unordered_map>
#include <map>

struct HierLBTypes {
  using hier_objid_t = int64_t;
  using hier_bin_t = int;
  using cont_hier_bin_t = std::list<hier_objid_t>;
  using cont_hier_objid_t = std::map<hier_bin_t, cont_hier_bin_t>;
};

#endif /*_HIERLB_TYPES_H_*/

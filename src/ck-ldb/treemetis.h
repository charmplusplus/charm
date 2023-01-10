#ifndef METISLB_H
#define METISLB_H

#include "TreeStrategyBase.h"
#include <metis.h>

namespace TreeStrategy
{
template <typename O, typename P, typename S>
class Metis : public Strategy<O, P, S>
{
public:
  Metis() = default;
  void solve(std::vector<O>& objs, std::vector<P>& procs, S& solution, bool objsSorted)
  {
    // Input fields:
    idx_t nvtxs = objs.size();
    idx_t ncon = O::dimension;
    std::vector<idx_t> xadj(nvtxs + 1);
    constexpr idx_t* adjncy = nullptr;
    std::vector<idx_t> vwgt(nvtxs * O::dimension);
    constexpr idx_t* vsize = nullptr;
    constexpr idx_t* adjwgt = nullptr;
    idx_t nparts = procs.size();
    constexpr real_t* tpwgts = nullptr;
    std::array<real_t, O::dimension> ubvec;
    ubvec.fill(1.0001);
    std::array<idx_t, METIS_NOPTIONS> options;
    METIS_SetDefaultOptions(options.data());
    // C style numbering
    options[METIS_OPTION_NUMBERING] = 0;
    // options[METIS_OPTION_PTYPE] = METIS_PTYPE_RB;

    // Output fields:
    // number of edges cut by the partitioning or total comm volume
    idx_t objval;
    // mapping of objs to partitions
    std::vector<idx_t> part(nvtxs);

    std::array<LoadFloatType, O::dimension> maxLoad;
    /** the object load is normalized to an integer between 0 and 1024 */
    for (const auto& obj : objs)
    {
      for (int i = 0; i < O::dimension; i++)
        maxLoad[i] = std::max(maxLoad[i], obj.getLoad(i));
    }
    for (int i = 0; i < objs.size(); i++)
    {
      auto* scaledLoads = &(vwgt[i * O::dimension]);
      for (int j = 0; j < O::dimension; j++)
      {
        scaledLoads[j] = 1024.0 * objs[i].getLoad(j) / maxLoad[j];
      }
    }

    // numVertices: num vertices in the graph; ncon: num balancing constrains
    // xadj, adjncy: of size n+1 and adjncy of 2m, adjncy[xadj[i]] through and
    // including adjncy[xadj[i+1]-1];
    // vwgt: weight of the vertices; vsize: amt of data that needs to be sent
    // for ith vertex is vsize[i]
    // adjwght: the weight of edges; numPes: total parts
    // tpwghts: target partition weight, can pass NULL to equally divide
    // ubvec: of size ncon to indicate allowed load imbalance tolerance (> 1.0)
    // options: array of options; edgecut: stores the edgecut; pemap: mapping
    METIS_PartGraphRecursive(&nvtxs, &ncon, xadj.data(), adjncy, vwgt.data(), vsize,
                             adjwgt, &nparts, tpwgts, ubvec.data(), options.data(), &objval,
                             part.data());

    for (int i = 0; i < nvtxs; i++)
    {
      solution.assign(objs[i], procs[part[i]]);
    }
  }
};

}  // namespace TreeStrategy

#endif /* METISLB_H */

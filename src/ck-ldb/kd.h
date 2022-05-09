#ifndef KD_H
#define KD_H

#include "TreeStrategyBase.h"
#include "kdnode.h"

namespace TreeStrategy
{
template <typename O, typename P, typename S, typename T>
class BaseKdLB : public Strategy<O, P, S>
{
public:
  BaseKdLB() = default;
  void solve(std::vector<O>& objs, std::vector<P>& procs, S& solution, bool objsSorted)
  {
    // Sorts by maxload in vector
    if (!objsSorted) std::sort(objs.begin(), objs.end(), CmpLoadGreater<O>());

    auto objsIter = objs.begin();
    T* tree = nullptr;
    for (int i = 0; i < procs.size() && objsIter != objs.end(); i++, objsIter++)
    {
      solution.assign(*objsIter, procs[i]);
      tree = T::insert(tree, procs[i]);
    }

    for (; objsIter != objs.end(); objsIter++)
    {
      auto proc = *(T::findMinNorm(tree, *objsIter));
      tree = T::remove(tree, proc);
      solution.assign(*objsIter, proc);
      tree = T::insert(tree, proc);
    }
  }
};

template <typename O, typename P, typename S>
class Kd : public BaseKdLB<O, P, S, RKDNode<P>>
{
};

template <typename O, typename P, typename S, typename T>
class BaseKdConstraint : public Strategy<O, P, S>
{
private:
  std::vector<KDFloatType> constraints;

public:
  BaseKdConstraint(json& config)
  {
    if (config.contains("constraints"))
    {
      constraints = config["constraints"].get<std::vector<KDFloatType>>();
    }
    else
    {
      CkPrintf("[KdConstraint] Warning: no constraints provided!\n");
    }
  }

  void solve(std::vector<O>& objs, std::vector<P>& procs, S& solution, bool objsSorted)
  {
    const int numConstraints = constraints.size();

    if (_lb_args.debug() > 1)
    {
      CkPrintf("[KdConstraint] Constraints:");
      for (const auto& value : constraints)
      {
        CkPrintf(" %f", value);
      }
      CkPrintf("\n");
    }


    // TODO: Sort only by non-constraint dimensions
    // Sorts by maxload in vector
    if (!objsSorted) std::sort(objs.begin(), objs.end(), CmpLoadGreater<O>());

    auto objsIter = objs.begin();
    T* tree = nullptr;
    for (int i = 0; i < procs.size() && objsIter != objs.end(); i++, objsIter++)
    {
      solution.assign(*objsIter, procs[i]);
      tree = T::insert(tree, procs[i], numConstraints);
    }

    for (; objsIter != objs.end(); objsIter++)
    {
      auto proc = *(T::findMinNormConstraints(tree, *objsIter, constraints));

      tree = T::remove(tree, proc);
      solution.assign(*objsIter, proc);
      tree = T::insert(tree, proc, numConstraints);
    }
  }
};

template <typename O, typename P, typename S>
class KdConstraint : public BaseKdConstraint<O, P, S, RKDNode<P>>
{
public:
  using BaseKdConstraint<O, P, S, RKDNode<P>>::BaseKdConstraint;
};


}  // namespace TreeStrategy

#endif /* KD_H */

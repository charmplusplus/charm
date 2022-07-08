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
  std::vector<bool> duringMigration;

  void assign(O& obj, P& proc, S& solution, const bool considerMigration)
  {
    solution.assign(obj, proc);

    // If this object is staying on its old PE, subtract out the constrained
    // dimensions that we are considering during migration. These values were
    // already added to the PE load once in the prep phase of solve, and then
    // again when the object was assigned above, so subtract them out.
    if (proc.id == obj.oldPe && considerMigration)
    {
      const auto numConstraints = constraints.size();
      for (int i = O::dimension - numConstraints; i < O::dimension; i++)
      {
        if (duringMigration[i - (O::dimension - numConstraints)])
          proc.unassignDim(obj, i);
      }
    }

    // Else, if the object is moving, do nothing, as those dimensions should
    // be counted on the old PE and on the new PE
  }

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

    if (config.contains("constrain_during_migration"))
    {
      CkAssert(duringMigration.size() == constraints.size());
      duringMigration = config["constrain_during_migration"].get<std::vector<bool>>();
    }
    else
    {
      // if not provided, default to all false
      duringMigration.resize(constraints.size(), false);
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
      CkPrintf("[KdConstraint] Constrain during migration:");
      for (const auto& value : duringMigration)
      {
        CkPrintf(" %d", value);
      }
      CkPrintf("\n");
    }

    // TODO: Sort only by non-constraint dimensions
    // Sorts by maxload in vector
    if (!objsSorted) std::sort(objs.begin(), objs.end(), CmpLoadGreater<O>());

    // If some value in duringMigration is true, then we must ensure that the constraint
    // is not violated even during migrations, meaning we must constrain the sum of
    // existing object + incoming objects in that dimension.
    const bool considerMigration = std::any_of(
        duringMigration.begin(), duringMigration.end(), [](const bool b) { return b; });
    if (considerMigration)
    {
      std::vector<int> procMap(CkNumPes(), -1);
      for (int i = 0; i < procs.size(); i++) procMap[procs[i].id] = i;
      for (const auto& o : objs)
      {
        auto& oldPe = procs[procMap[o.oldPe]];

        for (int i = O::dimension - numConstraints; i < O::dimension; i++)
        {
          if (duringMigration[i - (O::dimension - numConstraints)])
            oldPe.assignDim(o, i);
        }
      }
    }

    auto objsIter = objs.begin();
    T* tree = nullptr;
    for (int i = 0; i < procs.size() && objsIter != objs.end(); i++, objsIter++)
    {
      assign(*objsIter, procs[i], solution, considerMigration);
      tree = T::insert(tree, procs[i], numConstraints);
    }

    for (; objsIter != objs.end(); objsIter++)
    {
      auto proc = *(T::findMinNormConstraints(tree, *objsIter, constraints));

      tree = T::remove(tree, proc);
      assign(*objsIter, proc, solution, considerMigration);
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

#ifndef TREESTRATEGYFACTORY_H
#define TREESTRATEGYFACTORY_H

#include "TreeStrategyBase.h"
#include "greedy.h"
#include "refine.h"

namespace TreeStrategy
{
#define LB_STRATEGIES_FOR_TESTING 1

constexpr auto LBNames = {"Greedy",
                          "GreedyRefine",
                          "RefineA",
                          "RefineB",
                          "Random",
#if LB_STRATEGIES_FOR_TESTING
                          "Dummy",
                          "Rotate",
#endif
};

std::string getLBNamesString()
{
  std::ostringstream output;
  for (const auto& name : LBNames)
  {
    output << "\n\t" << name;
  }
  return output.str();
}

class Factory
{
public:
  // NOTE: This is the only place currently where the templates for each strategy
  // are instantiated. This means that code for any strategies that are disabled here
  // during preprocessing will not be part of the executable (because the templates
  // won't be instantiated)
  template <class O, class P, class S>
  static Strategy<O, P, S>* makeStrategy(const std::string& name, json& config)
  {
    if (name == "Greedy") return new Greedy<O, P, S>();
    if (name == "GreedyRefine") return new GreedyRefine<O, P, S>(config);
    if (name == "RefineA") return new RefineA<O, P, S>();
    if (name == "RefineB") return new RefineB<O, P, S>();
    if (name == "Random") return new Random<O, P, S>();
#if LB_STRATEGIES_FOR_TESTING
    if (name == "Dummy") return new Dummy<O, P, S>();
    if (name == "Rotate") return new Rotate<O, P, S>();
#endif
    std::string error_msg("Unrecognized strategy ");
    error_msg += name;
    CkAbort("%s\n", error_msg.c_str());
  }
};
}  // namespace TreeStrategy
#endif /* TREESTRATEGYFACTORY_H */

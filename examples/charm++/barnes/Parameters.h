#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include "defines.h"
#include "charm++.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <map>

#include "pup_stl.h"

using namespace std;

struct Parameters {
  string filename;

  Real theta;
  Real dtime;
  Real dthf;
  Real epssq;
  Real tolsq;

  int numTreePieces;
  int numParticles;
  int ppc;
  int ppb;

  int yieldPeriod;
  int cacheLineSize;

  int iterations;

  // Load balancing. lbPeriod <= 0 disables AtSync entirely; otherwise the
  // balancer runs after iteration firstLbIteration and every lbPeriod
  // iterations after that.
  int firstLbIteration;
  int lbPeriod;
  // Split the AtSync barrier (needs +LBAsync). The tree pieces report the end
  // of an iteration as soon as they have joined the step, so the DataManager
  // decomposes while the strategy runs and elements move, and it waits for the
  // step only where it needs the elements to be still.
  int asyncLb;
  // Iterations between the two halves of the split barrier. The element joins
  // the step, keeps iterating for this many iterations, and only then parks in
  // AtSyncWait(). What that buys is the strategy: DiffusionLB's rounds run
  // while the application works, instead of inside one decomposition's worth
  // of slack.
  //
  // It does NOT buy migration overlap, and that is a property of this
  // application rather than of the runtime. Under +LBAsync an element is
  // normally movable at any entry-method boundary it has declared safe --
  // leanmd's Computes carry no ReadyMigrate at all and are moved whenever the
  // strategy decides. A tree piece cannot be, because the decomposition binds
  // it to its PE's DataManager for a whole iteration: publishTreePieceMap()
  // reduces the tree-piece-to-PE map once per iteration, every sender
  // addresses its particle blocks by it, and assembleReceivedBlocks() then
  // attributes what arrives through a raw local TreePiece*. An element that
  // moves after the map is published has its particles delivered to a PE that
  // no longer hosts it, where descr.owner is null. So the safe-to-pack window
  // stays closed across the lag and the moves happen at the park. Opening it
  // mid-lag needs the blocks to follow the element -- forwarding plus new
  // termination detection -- which is a decomposition change, not a flag.
  int lbLag;
  // How many iterations before a balancing iteration the measurement window
  // opens. Instrumentation is off outside it, so the strategy reads a short,
  // recent window: measured entirely after the previous step's migrations
  // settled, and ending at the decision. 0 instruments continuously.
  int lbWindow;
  // Levels a single histogram round may refine an over-full bin by. One is the
  // original behaviour. See DataManager::receiveHistogram.
  int decompLevels;
  // Stage 7. Run the LOCAL half of the traversal on the device instead of the
  // host. The remote half stays on the host either way: the data it needs is
  // not resident here until push-based LET lands.
  int deviceWalk;
  // Push a locally essential tree to every other PE before the traversal,
  // instead of pulling remote nodes one round trip at a time.
  int useLet;
  // Send the decomposition's particles straight from device memory instead of
  // copying them through the host on both ends.
  int deviceExchange;
  // Splice pushed trees into the device tree and let one device walk cover
  // both halves, instead of walking the remote half on the host.
  int deviceLet;
  // Initial tree piece placement. 0 = the default round-robin map, 1 = a block
  // map, which is what makes this benchmark imbalanced; see BlockMap.
  int blockMap;
  // Consecutive tree piece indices per node before the placement moves on.
  int mapChunk;
  // Opt in to the node-aware block cyclic placement. Off by default: the
  // stock Charm++ map is what runs unless asked otherwise.
  int mapCyclic;

  // Interaction-list size, in sources, at which a tree piece stops
  // accumulating and launches. Only a memory bound: at typical particle
  // counts per PE a tree piece's whole list is smaller than this and it
  // launches once, at the end of its traversals.
  int gpuFlushLimit;

  // Arm the quiescence-based deadlock detector (on by default).
  int quiescenceCheck;

  //int branchFactor;

  void pup(PUP::er &p){
    p | filename;
    p | numTreePieces;
    p | numParticles;
    p | dtime;
    p | dthf;
    p | tolsq;
    p | epssq;
    p | ppc;
    p | ppb;
    p | yieldPeriod;
    p | theta;
    p | cacheLineSize;
    p | iterations;
    p | firstLbIteration;
    p | lbPeriod;
    p | asyncLb;
    p | lbLag;
    p | lbWindow;
    p | decompLevels;
    p | deviceWalk;
    p | useLet;
    p | deviceExchange;
    p | deviceLet;
    p | blockMap;
    p | mapChunk;
    p | mapCyclic;
    p | gpuFlushLimit;
    p | quiescenceCheck;
  }

  void extractParameters(int argc, char **argv, map<string,string> &tab){
    for(int i = 0; i < argc; i++){
      string arg = string(argv[i]);
      size_t pos = arg.find("=");
      if(pos != string::npos){
        size_t len = arg.length();
        string key = arg.substr(1,pos-1); 
        string val = arg.substr(pos+1,len-pos-1);
        tab[key] = val;
      }
    }
  }

  string getparam(string name, map<string,string> &table)
  {
    map<string,string>::iterator it = table.find(name);
    if(it != table.end()){
      return it->second;
    }
    return string();
  }

  /*
   * GETIPARAM, ..., GETDPARAM: get int, long, bool, or double parameters.
   */

  int getiparam(string name, int def, map<string,string> &tab)
  {
    string val;

    val = getparam(name,tab);
    if(val.empty())
      return def;
    else
      return (atoi(val.c_str()));
  }

  long getlparam(string name, map<string,string> &tab)
  {
    string val;

    val = getparam(name,tab);
    if(val.empty())
      return -1;
    else 
      return (atol(val.c_str()));
  }

  bool getbparam(string name, map<string,string> &tab)
  {
    string val;

    val = getparam(name,tab);
    if (strchr("tTyY1", *(val.c_str())) != 0) {
      return (true);
    }
    if (strchr("fFnN0", *(val.c_str())) != 0) {
      return (false);
    }
    fprintf(stderr,"getbparam: %s=%s not bool\n", name.c_str(), val.c_str());
    return false;
  }

  Real getrparam(string name, Real default_value, map<string,string> &tab)
  {
    string val;

    val = getparam(name,tab);
    if(val.empty())
      return default_value;
    else 
      return (atof(val.c_str()));
  }

  /*
   * EXTRVALUE: extract value from name=value string.
   */

  string getsparam(string arg, map<string,string> &tab)
  {
    return getparam(arg,tab);
  }


};

// Is `iter` an iteration at which the balancer runs? Shared so that the
// DataManager's instrumentation window and the tree pieces' AtSync schedule
// are driven by one definition rather than two that have to agree.
inline bool isBalancingIteration(const Parameters &p, int iter){
  if(p.lbPeriod <= 0) return false;
  if(iter < p.firstLbIteration) return false;
  if(iter >= p.iterations) return false;
  return ((iter - p.firstLbIteration) % p.lbPeriod) == 0;
}

#endif

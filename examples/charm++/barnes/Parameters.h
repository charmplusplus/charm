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
  // How many iterations before a balancing iteration the measurement window
  // opens. Instrumentation is off outside it, so the strategy reads a short,
  // recent window: measured entirely after the previous step's migrations
  // settled, and ending at the decision. 0 instruments continuously.
  int lbWindow;
  // Initial tree piece placement. 0 = the default round-robin map, 1 = a block
  // map, which is what makes this benchmark imbalanced; see BlockMap.
  int blockMap;

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
    p | lbWindow;
    p | blockMap;
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

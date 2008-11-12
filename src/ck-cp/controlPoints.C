#include <charm++.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <vector>
#include <utility>
#include <sys/time.h>

#include "ControlPoints.decl.h"

#include "LBDatabase.h"
#include "controlPoints.h"

using namespace std;

#define CONTROL_POINT_PERIOD 8000
#define OPTIMIZER_TRANSITION 5

static void periodicProcessControlPoints(void* ptr, double currWallTime);


// A pointer to this PE's controlpoint manager Proxy
/* readonly */ CProxy_controlPointManager localControlPointManagerProxy;
/* readonly */ int random_seed;


class instrumentedPhase {
public:
  std::map<string,int> controlPoints; // The control point values for this phase(don't vary within the phase)
  std::multiset<double> times;  // A list of times observed for iterations in this phase
  
  int memoryUsageMB;
  
  instrumentedPhase(){
    memoryUsageMB = -1;
  }
  
  void clear(){
    controlPoints.clear();
    times.clear();
  }

  // Provide a previously computed value, or a value from a previous run
  bool haveValueForName(const char* name){
    string n(name);
    return (controlPoints.count(n)>0);
  }

  void operator=(const instrumentedPhase& p){
    controlPoints = p.controlPoints;
    times = p.times;
    memoryUsageMB = p.memoryUsageMB;
  }



  bool operator<(const instrumentedPhase& p){
    CkAssert(hasSameKeysAs(p)); 
    std::map<string,int>::iterator iter1 = controlPoints.begin();
    std::map<string,int>::const_iterator iter2 = p.controlPoints.begin();
    for(;iter1 != controlPoints.end() && iter2 != p.controlPoints.end(); iter1++, iter2++){
      if(iter1->second < iter2->second){
	return true;
      }
    }
    return false;
  }


  // Determines if the control point values and other information exists
  bool hasValidControlPointValues(){
    std::map<string,int>::iterator iter;
    for(iter = controlPoints.begin();iter != controlPoints.end(); iter++){
      if(iter->second == -1){ 
        return false; 
      }  
    }

    return true;
  }



  bool operator==(const instrumentedPhase& p){
    CkAssert(hasSameKeysAs(p));
    std::map<string,int>::iterator iter1 = controlPoints.begin();
    std::map<string,int>::const_iterator iter2 = p.controlPoints.begin();
    for(;iter1 != controlPoints.end() && iter2 != p.controlPoints.end(); iter1++, iter2++){ 
      if(iter1->second != iter2->second){ 
        return false; 
      }  
    }
    return true;
  }

  /// Verify the names of the control points are consistent
  /// note: std::map stores the pairs in a sorted order based on their first component 
  bool hasSameKeysAs(const instrumentedPhase& p){
    
    if(controlPoints.size() != p.controlPoints.size())
      return false;

    std::map<string,int>::iterator iter1 = controlPoints.begin(); 
    std::map<string,int>::const_iterator iter2 = p.controlPoints.begin(); 

    for(;iter1 != controlPoints.end() && iter2 != p.controlPoints.end(); iter1++, iter2++){  
      if(iter1->first != iter2->first)
	return false;
    } 

    return true; 
  }


  void addAllNames(std::set<string> names_) {
    
    std::set<string> names = names_;
    
    // Remove all the names that we already have
    std::map<std::string,int>::iterator iter;
    
    for(iter = controlPoints.begin(); iter != controlPoints.end(); iter++){
      names.erase(iter->first);
    }
    
    // Add -1 values for each name we didn't find
    std::set<std::string>::iterator iter2;
    for(iter2 = names.begin(); iter2 != names.end(); iter2++){
      controlPoints.insert(make_pair(*iter2,-1));
      CkPrintf("One of the datasets was missing a value for %s, so -1 was used\n", iter2->c_str());
    }

  }



  void print() {
    std::map<std::string,int>::iterator iter;

    if(controlPoints.size() == 0){
      CkPrintf("no control point values found\n");
    }
    
    for(iter = controlPoints.begin(); iter != controlPoints.end(); iter++){
      std::string name = iter->first;
      int val = iter->second;
      CkPrintf("%s ---> %d\n",  name.c_str(),  val);
    } 
    
  }
  
  
};



class instrumentedData {
public:
  std::vector<instrumentedPhase> phases;

  /// get control point names for all phases
  std::set<string> getNames(){
    std::set<string> names;
    
    std::vector<instrumentedPhase>::iterator iter;
    for(iter = phases.begin();iter!=phases.end();iter++) {
      
      std::map<string,int>::iterator iter2;
      for(iter2 = iter->controlPoints.begin(); iter2 != iter->controlPoints.end(); iter2++){
	names.insert(iter2->first);
      }
      
    }  
    return names;

  } 


  void cleanupNames(){
    std::set<string> names = getNames();
    
    std::vector<instrumentedPhase>::iterator iter;
    for(iter = phases.begin();iter!=phases.end();iter++) {
      iter->addAllNames(names);
    }
  }


  /// Remove one phase with invalid control point values if found
  bool filterOutOnePhase(){
    // Note: calling erase on a vector will invalidate any iterators beyond the deletion point
    std::vector<instrumentedPhase>::iterator iter;
    for(iter = phases.begin(); iter != phases.end(); iter++) {
      if(! iter->hasValidControlPointValues()  || iter->times.size()==0){
	// CkPrintf("Filtered out a phase with incomplete control point values\n");
	phases.erase(iter);
	return true;
      } else {
	//	CkPrintf("Not filtering out some phase with good control point values\n");
      }
    }
    return false;
  }
  
  /// Drop any phases that do not contain timings or control point values
  void filterOutIncompletePhases(){
    bool done = false;
    while(filterOutOnePhase()){
      // do nothing
    }
  }


  string toString(){
    ostringstream s;

    verify();

    filterOutIncompletePhases();

    // HEADER:
    s << "# HEADER:\n";
    s << "# Data for use with Isaac Dooley's Control Point Framework\n";
    s << string("# Number of instrumented timings in this file:\n"); 
    s << phases.size() << "\n" ;
    
    if(phases.size() > 0){
      
      std::map<string,int> &ps = phases[0].controlPoints; 
      std::map<string,int>::iterator cpiter;

      // SCHEMA:
      s << "# SCHEMA:\n";
      s << "# number of named control points:\n";
      s << ps.size() << "\n";
      
      for(cpiter = ps.begin(); cpiter != ps.end(); cpiter++){
	s << cpiter->first << "\n";
      }
      
      // DATA:
      s << "# DATA:\n";
      s << "# first field is memory usage (MB). Then there are the " << ps.size()  << " control points values, followed by one or more timings" << "\n";
      s << "# number of control point sets: " << phases.size() << "\n";
      std::vector<instrumentedPhase>::iterator runiter;
      for(runiter=phases.begin();runiter!=phases.end();runiter++){

	// Print the memory usage
	 s << runiter->memoryUsageMB << "    "; 

	// Print the control point values
	for(cpiter = runiter->controlPoints.begin(); cpiter != runiter->controlPoints.end(); cpiter++){ 
	  s << cpiter->second << " "; 
	}

	s << "     ";

	// Print the times
	std::multiset<double>::iterator titer;
	for(titer = runiter->times.begin(); titer != runiter->times.end(); titer++){
	  s << *titer << " ";
	}

	s << "\n";
	
      }
 
    }

    return s.str();
    
  }


  /// Verify that all our phases of data have the same sets of control point names
  void verify(){
    if(phases.size() > 1){
      instrumentedPhase & firstpoint = phases[0];
      std::vector<instrumentedPhase>::iterator iter;
      for(iter = phases.begin();iter!=phases.end();iter++){
	CkAssert( firstpoint.hasSameKeysAs(*iter));
      }  
    } 
  }


  // Find the fastest time from previous runs
  instrumentedPhase findBest(){
    CkAssert(phases.size()>0);

    double total_time = 0.0; // total for all times
    int total_count = 0;

    instrumentedPhase best_phase;
    double best_phase_avgtime = std::numeric_limits<double>::max();

    int valid_phase_count = 0;

    std::vector<instrumentedPhase>::iterator iter;
    for(iter = phases.begin();iter!=phases.end();iter++){
      if(iter->hasValidControlPointValues()){
	valid_phase_count++;

	double total_for_phase = 0.0;
	int phase_count = 0;

	// iterate over all times for this control point configuration
	std::multiset<double>::iterator titer;
	for(titer = iter->times.begin(); titer != iter->times.end(); titer++){
	  total_count++;
	  total_time += *titer;
	  total_for_phase += *titer;
	  phase_count ++;
	}

	double phase_average_time = total_for_phase / (double)phase_count;

	if(phase_average_time  < best_phase_avgtime){
	  best_phase = *iter;
	  best_phase_avgtime = phase_average_time; 
	}

      }
    }
    
    CkAssert(total_count > 0);

    double avg = total_time / total_count;

    if(CkMyPe() == 0){
      CkPrintf("Best average time for a phase was %.1lf\n", best_phase_avgtime);
      CkPrintf("Mean time for all %d times in the %d valid recorded phases was %.1lf\n", total_count, valid_phase_count, avg );
    }

    // Compute standard deviation
    double sumx=0.0;
    for(iter = phases.begin();iter!=phases.end();iter++){
      if(iter->hasValidControlPointValues()){
	std::multiset<double>::iterator titer;
	for(titer = iter->times.begin(); titer != iter->times.end(); titer++){
	  sumx += (avg - *titer)*(avg - *titer);
	} 
      }
    }
    
    double std_dev = sqrt(sumx / total_count);

    if(CkMyPe() == 0){
      CkPrintf("Standard Deviation for previous runs was %.2lf   or %.1lf%% of mean\n", std_dev, std_dev/avg*100.0);
      CkPrintf("The best phase average time was %.1lf%% faster than the mean\n", (avg-best_phase_avgtime)/avg*100.0);

    }

    return best_phase;
  }
  
};



/// Return an integer between 0 and num-1 inclusive
/// If different seed, name, and random_seed values are provided, the returned values are pseudo-random
unsigned int randInt(unsigned int num, const char* name, int seed=0){
  CkAssert(num > 0);
  CkAssert(num < 1000);

  unsigned long hash = 0;
  unsigned int c;
  unsigned char * str = (unsigned char*)name;

  while (c = *str++){
    unsigned int c2 = (c+64)%128;
    unsigned int c3 = (c2*5953)%127;
    hash = c3 + (hash << 6) + (hash << 16) - hash;
  }

  unsigned long shuffled1 = (hash*2083)%7907;
  unsigned long shuffled2 = (seed*4297)%2017;

  unsigned long shuffled3 = (random_seed*4799)%7919;

  unsigned int namehash = shuffled3 ^ shuffled1 ^ shuffled2;

  unsigned int result = ((namehash * 6029) % 1117) % num;

  CkAssert(result >=0 && result < num);
  return result;
}




//=============================================================================
// THE MODULE CODE IS HERE: 
//=============================================================================


// A group with representatives on all nodes
class controlPointManager : public CBase_controlPointManager {
public:
  
  char * dataFilename;
  
  instrumentedData allData;
  instrumentedPhase thisPhaseData;
  instrumentedPhase best_phase;
  
  std::map<string, pair<int,int> > controlPointSpace;

  std::set<string> staticControlPoints;


  CkCallback granularityCallback;
  bool haveGranularityCallback;
  bool frameworkShouldAdvancePhase;
  
  int phase_id;

  bool alreadyRequestedMemoryUsage;

  controlPointManager(){
    alreadyRequestedMemoryUsage = false;
    
    dataFilename = (char*)malloc(128);
    sprintf(dataFilename, "controlPointData.txt");
    
    frameworkShouldAdvancePhase = false;
    haveGranularityCallback = false;
    CkPrintf("[%d] controlPointManager() Constructor Initializing control points, and loading data file\n", CkMyPe());
    
    phase_id = 0;
    
    localControlPointManagerProxy = thisProxy;
    
    loadDataFile();
    
    if(allData.phases.size()>0){
      allData.findBest();
    }
    
    if(CkMyPe() == 0)
      CcdCallFnAfterOnPE((CcdVoidFn)periodicProcessControlPoints, (void*)NULL, CONTROL_POINT_PERIOD, CkMyPe());

  }
  
  ~controlPointManager(){
    CkPrintf("[%d] controlPointManager() Destructor\n", CkMyPe());
  }


  /// Loads the previous run data file
  void loadDataFile(){
    ifstream infile(dataFilename);
    vector<string> names;
    string line;
  
    while(getline(infile,line)){
      if(line[0] != '#')
	break;
    }
  
    int numTimings = 0;
    istringstream n(line);
    n >> numTimings;
  
    while(getline(infile,line)){ 
      if(line[0] != '#') 
	break; 
    }

    int numControlPointNames = 0;
    istringstream n2(line);
    n2 >> numControlPointNames;
  
    for(int i=0; i<numControlPointNames; i++){
      getline(infile,line);
      names.push_back(line);
    }

    for(int i=0;i<numTimings;i++){
      getline(infile,line);
      while(line[0] == '#')
	getline(infile,line); 

      instrumentedPhase ips;

      istringstream iss(line);

      // Read memory usage for phase
      iss >> ips.memoryUsageMB;
      //     CkPrintf("Memory usage loaded from file: %d\n", ips.memoryUsageMB);


      // Read control point values
      for(int cp=0;cp<numControlPointNames;cp++){
	int cpvalue;
	iss >> cpvalue;
	ips.controlPoints.insert(make_pair(names[cp],cpvalue));
      }

      double time;

      while(iss >> time){
	ips.times.insert(time);
#if DEBUG > 5
	CkPrintf("read time %lf from file\n", time);
#endif
      }

      allData.phases.push_back(ips);

    }

    infile.close();
  }



  /// Add the current data to allData and output it to a file
  void writeDataFile(){
    CkPrintf("============= writeDataFile() ============\n");
    ofstream outfile(dataFilename);
    allData.phases.push_back(thisPhaseData);
    allData.cleanupNames();
    outfile << allData.toString();
    outfile.close();
  }


  void setGranularityCallback(CkCallback cb, bool _frameworkShouldAdvancePhase){
    frameworkShouldAdvancePhase = _frameworkShouldAdvancePhase;
    granularityCallback = cb;
    haveGranularityCallback = true;
  }

  /// Called periodically by the runtime to handle the control points
  /// Currently called on each PE
  void processControlPoints(){
#if DEBUG
    CkPrintf("[%d] processControlPoints()\n", CkMyPe());
#endif
    if(haveGranularityCallback){
      if(frameworkShouldAdvancePhase){
	gotoNextPhase();	
      }
      
      controlPointMsg *msg = new(0) controlPointMsg;
      granularityCallback.send(msg); 
    }


    if(CkMyPe() == 0 && !alreadyRequestedMemoryUsage){
      alreadyRequestedMemoryUsage = true;
      CkCallback *cb = new CkCallback(CkIndex_controlPointManager::gatherMemoryUsage(NULL), 0, thisProxy);
      //      thisProxy.requestMemoryUsage(*cb);
      delete cb;
    }

  }
  
  instrumentedPhase *previousPhaseData(){
    int s = allData.phases.size();
    if(s >= 1 && phase_id > 0) {
      return &(allData.phases[s-1]);
    } else {
      return NULL;
    }
  }
  
  
  void gotoNextPhase(){

    LBDatabase * myLBdatabase = LBDatabaseObj();
    LBDB * myLBDB = myLBdatabase->getLBDB();       // LBDB is Defined in LBDBManager.h
    const CkVec<LBObj*> objs = myLBDB->getObjs();
    const int objCount = myLBDB->getObjCount();

    CkPrintf("LBDB info: objCount=%d objs contains %d LBObj* \n", objCount, objs.length());

    floatType maxObjWallTime = -1.0;
    
    for(int i=0;i<objs.length();i++){
      LBObj* o = objs[i];
      const LDObjData d = o->ObjData();
      floatType cpuTime = d.cpuTime;
      floatType wallTime = d.wallTime;
      // can also get object handles from the LDObjData struct
      CkPrintf("[%d] LBDB Object[%d]: cpuTime=%f wallTime=%f\n", CkMyPe(), i, cpuTime, wallTime);
      if(wallTime > maxObjWallTime){

      }
      
    }

    myLBDB->ClearLoads(); // BUG: Probably very dangerous if we are actually using load balancing
    
    
    
    
    // increment phase id
    phase_id++;
    
    CkPrintf("Now in phase %d\n", phase_id);
    
    // save the timing information from this phase
    allData.phases.push_back(thisPhaseData);
        
    // clear the timing information that will be used for the next phase
    thisPhaseData.clear();
    
  }

  // The application can set the instrumented time for this phase
  void setTiming(double time){
     thisPhaseData.times.insert(time);
  }
  


  void requestMemoryUsage(CkCallback cb){
    int m = CmiMaxMemoryUsage() / 1024 / 1024;
    CmiResetMaxMemory();
    CkPrintf("PE %d Memory Usage is %d MB\n",CkMyPe(), m);
    contribute(sizeof(int),&m,CkReduction::max_int, cb);
  }

  void gatherMemoryUsage(CkReductionMsg *msg){
    int size=msg->getSize() / sizeof(int);
    CkAssert(size==1);
    int *m=(int *) msg->getData();

    CkPrintf("[%d] Max Memory Usage for all processors is %d MB\n", CkMyPe(), m[0]);
    
    instrumentedPhase* prevPhase = previousPhaseData();
    if(prevPhase != NULL){
      prevPhase->memoryUsageMB = m[0];
    } else {
      CkPrintf("No place to store memory usage");
    }

    alreadyRequestedMemoryUsage = false;
    delete msg;
  }



};



void gotoNextPhase(){
  localControlPointManagerProxy.ckLocalBranch()->gotoNextPhase();
}


// A mainchare that is used just to create our controlPointManager group at startup
class controlPointMain : public CBase_controlPointMain {
public:
  controlPointMain(CkArgMsg* args){
    struct timeval tp;
    gettimeofday(& tp, NULL);
    random_seed = (int)tp.tv_usec ^ (int)tp.tv_sec;

    localControlPointManagerProxy = CProxy_controlPointManager::ckNew();
  }
  ~controlPointMain(){}
};

void registerGranularityChangeCallback(CkCallback cb, bool frameworkShouldAdvancePhase){
  CkAssert(CkMyPe() == 0);
  CkPrintf("registerGranularityChangeCallback\n");
  localControlPointManagerProxy.ckLocalBranch()->setGranularityCallback(cb, frameworkShouldAdvancePhase);
}


void registerControlPointTiming(double time){
  CkAssert(CkMyPe() == 0);
#if DEBUG>0
  CkPrintf("Program registering its own timing with registerControlPointTiming(time=%lf)\n", time);
#endif
  localControlPointManagerProxy.ckLocalBranch()->setTiming(time);
}

extern "C" void controlPointShutdown(){
  CkAssert(CkMyPe() == 0);
  CkPrintf("[%d] controlPointShutdown() at CkExit()\n", CkMyPe());

  localControlPointManagerProxy.ckLocalBranch()->writeDataFile();
  CkExit();
}


void controlPointInitNode(){
  CkPrintf("controlPointInitNode()\n");
  registerExitFn(controlPointShutdown);
}

static void periodicProcessControlPoints(void* ptr, double currWallTime){
#ifdef DEBUG
  CkPrintf("[%d] periodicProcessControlPoints()\n", CkMyPe());
#endif
  localControlPointManagerProxy.ckLocalBranch()->processControlPoints();
  CcdCallFnAfterOnPE((CcdVoidFn)periodicProcessControlPoints, (void*)NULL, CONTROL_POINT_PERIOD, CkMyPe());
}





// Static point for life of program: randomly chosen, no optimizer
int staticPoint(const char *name, int lb, int ub){
  instrumentedPhase &thisPhaseData = localControlPointManagerProxy.ckLocalBranch()->thisPhaseData;
  std::set<string> &staticControlPoints = localControlPointManagerProxy.ckLocalBranch()->staticControlPoints;  

  int result = lb + randInt(ub-lb+1, name);
  
  localControlPointManagerProxy.ckLocalBranch()->controlPointSpace.insert(std::make_pair(string(name),std::make_pair(lb,ub))); 
  thisPhaseData.controlPoints.insert(std::make_pair(string(name),result)); 
  staticControlPoints.insert(string(name));

  return result;
}



bool valueShouldBeProvidedByOptimizer(){
  
  const int effective_phase = localControlPointManagerProxy.ckLocalBranch()->allData.phases.size();
  const int phase_id = localControlPointManagerProxy.ckLocalBranch()->phase_id; 
  
  std::map<string, pair<int,int> > &controlPointSpace = localControlPointManagerProxy.ckLocalBranch()->controlPointSpace; 
  
  double spaceSize = 1.0;
  std::map<string, pair<int,int> >::iterator iter;
  for(iter = controlPointSpace.begin(); iter != controlPointSpace.end(); iter++){
    spaceSize *= iter->second.second - iter->second.first + 1;
  }

  //  CkPrintf("Control Point Space:\n\t\tnumber of control points = %d\n\t\tnumber of possible configurations = %.0lf\n", controlPointSpace.size(), spaceSize);

#if 1
  return effective_phase > 1 && phase_id > 1;
#else
  return effective_phase >= OPTIMIZER_TRANSITION && phase_id > 3;
#endif
}









int valueProvidedByOptimizer(const char * name){
  const int phase_id = localControlPointManagerProxy.ckLocalBranch()->phase_id;
  const int effective_phase = localControlPointManagerProxy.ckLocalBranch()->allData.phases.size();

  //#define OPTIMIZER_USE_BEST_TIME  
#if OPTIMIZER_USE_BEST_TIME  
  static bool firstTime = true;
  if(firstTime){
    firstTime = false;
    CkPrintf("Finding best phase\n");
    instrumentedPhase p = localControlPointManagerProxy.ckLocalBranch()->allData.findBest(); 
    CkPrintf("p=:\n");
    p.print();
    CkPrintf("\n");
    localControlPointManagerProxy.ckLocalBranch()->best_phase = p;
  }
  
  
  instrumentedPhase &p = localControlPointManagerProxy.ckLocalBranch()->best_phase;
  int result = p.controlPoints[std::string(name)];
  CkPrintf("valueProvidedByOptimizer(): Control Point \"%s\" for phase %d chosen out of best previous phase to be: %d\n", name, phase_id, result);
  return result;
#elsif SIMULATED_ANNEALING
  
  // Simulated Annealing style hill climbing method
  //
  // Find the best search space configuration, and try something
  // nearby it, with a radius decreasing as phases increase
  
  std::map<string, pair<int,int> > &controlPointSpace = localControlPointManagerProxy.ckLocalBranch()->controlPointSpace;  
  
  CkPrintf("Finding best phase\n"); 
  instrumentedPhase p = localControlPointManagerProxy.ckLocalBranch()->allData.findBest();  
  CkPrintf("best found:\n"); 
  p.print(); 
  CkPrintf("\n"); 
  
  int before = p.controlPoints[std::string(name)];   
  
  int minValue =  controlPointSpace[std::string(name)].first;
  int maxValue =  controlPointSpace[std::string(name)].second;
  
  int convergeByPhase = 100;
  
  // Determine from 0.0 to 1.0 how far along we are
  double progress = (double) min(effective_phase, convergeByPhase) / (double)convergeByPhase;
  
  int range = (maxValue-minValue+1)*(1.0-progress);

  CkPrintf("========================== Hill climbing progress = %lf  range=%d\n", progress, range); 

  int high = min(before+range, maxValue);
  int low = max(before-range, minValue);
  
  int result = low;

  if(high-low > 0){
    result += randInt(high-low, name, phase_id); 
  } 

  CkPrintf("valueProvidedByOptimizer(): Control Point \"%s\" for phase %d chosen by hill climbing to be: %d\n", name, phase_id, result); 
  return result; 

#else

  std::map<string, pair<int,int> > &controlPointSpace = localControlPointManagerProxy.ckLocalBranch()->controlPointSpace;
  std::set<string> &staticControlPoints = localControlPointManagerProxy.ckLocalBranch()->staticControlPoints;  
   
  int numDimensions = controlPointSpace.size();
  CkAssert(numDimensions > 0);
  
  int lowerBounds[numDimensions];
  int upperBounds[numDimensions]; 
  
  int d=0;
  std::map<string, pair<int,int> >::iterator iter;
  for(iter = controlPointSpace.begin(); iter != controlPointSpace.end(); iter++){
    //    CkPrintf("Examining dimension %d\n", d);

    string name = iter->first;
    if(staticControlPoints.count(name) >0 ){
      cout << " control point " << name << " is static " << endl;
    } else{
      cout << " control point " << name << " is not static " << endl;
    }

    lowerBounds[d] = iter->second.first;
    upperBounds[d] = iter->second.second;
    d++;
  }
   

  std::string s[numDimensions];
  d=0;
  for(std::map<string, pair<int,int> >::iterator niter=controlPointSpace.begin(); niter!=controlPointSpace.end(); niter++){
    s[d] = niter->first;
    // cout << "s[" << d << "]=" << s[d] << endl;
    d++;
  }
  
  
  // Create the first possible configuration
  int config[numDimensions+1]; // one value for each dimension and a
			       // final one to hold the carry
			       // producing an invalid config
  config[numDimensions] = 0;
  for(int i=0;i<numDimensions;i++){
    config[i] = lowerBounds[i];
  }
  
  // Increment until finding an unused configuration
  localControlPointManagerProxy.ckLocalBranch()->allData.cleanupNames(); // put -1 values in for any control points missing
  std::vector<instrumentedPhase> &phases = localControlPointManagerProxy.ckLocalBranch()->allData.phases;     

  while(true){
    
    std::vector<instrumentedPhase>::iterator piter; 
    bool testedConfiguration = false; 
    for(piter = phases.begin(); piter != phases.end(); piter++){ 
      
      // Test if the configuration matches this phase
      bool match = true;
      for(int j=0;j<numDimensions;j++){
	match &= piter->controlPoints[s[j]] == config[j];
      }
      
      if(match){
	testedConfiguration = true; 
	break;
      } 
    } 
    
    if(testedConfiguration == false){ 
      break;
    } 
    
    // go to next configuration
    config[0] ++;
    // Propagate "carrys"
    for(int i=0;i<numDimensions;i++){
      if(config[i] > upperBounds[i]){
	config[i+1]++;
	config[i] = lowerBounds[i];
      }
    }
    // check if we have exhausted all possible values
    if(config[numDimensions] > 0){
      break;
    }
       
  }
  
  if(config[numDimensions] > 0){
    for(int i=0;i<numDimensions;i++){ 
      config[i]= (lowerBounds[i]+upperBounds[i]) / 2;
    }
  }
  
  
  int result=-1;  

  std::string name_s(name);
  for(int i=0;i<numDimensions;i++){
    //    cout << "Comparing " << name_s <<  " with s[" << i << "]=" << s[i] << endl;
    if(name_s.compare(s[i]) == 0){
      result = config[i];
    }
  }

  CkAssert(result != -1);


  CkPrintf("valueProvidedByOptimizer(): Control Point \"%s\" for phase %d chosen by exhaustive search to be: %d\n", name, phase_id, result); 
  return result; 


#endif
  
}





#define isInRange(v,a,b) ( ((v)<=(a)&&(v)>=(b)) || ((v)<=(b)&&(v)>=(a)) )


// Dynamic point varies throughout the life of program
// The value it returns is based upon phase_id, a counter that changes for each phase of computation
int controlPoint2Pow(const char *name, int fine_granularity, int coarse_granularity){
  instrumentedPhase &thisPhaseData = localControlPointManagerProxy.ckLocalBranch()->thisPhaseData;
  const int phase_id = localControlPointManagerProxy.ckLocalBranch()->phase_id;

  int result;

  // Use best configuration after a certain point
  if(valueShouldBeProvidedByOptimizer()){
    result = valueProvidedByOptimizer(name);
  } 
  else {

    int l1 = log2l(fine_granularity);
    int l2 = log2l(coarse_granularity);
  
    if (l1 > l2){
      int tmp;
      tmp = l2;
      l2 = l1; 
      l1 = tmp;
    }
    CkAssert(l1 <= l2);
    
    int l = l1 + randInt(l2-l1+1,name, phase_id);

    result = 1 << l;

    CkAssert(isInRange(result,fine_granularity,coarse_granularity));
    CkPrintf("Control Point \"%s\" for phase %d chosen randomly to be: %d\n", name, phase_id, result);
  }

  thisPhaseData.controlPoints.insert(std::make_pair(string(name),result));

  return result;
}



int controlPoint(const char *name, int lb, int ub){
  instrumentedPhase &thisPhaseData = localControlPointManagerProxy.ckLocalBranch()->thisPhaseData;
  const int phase_id = localControlPointManagerProxy.ckLocalBranch()->phase_id;
  std::map<string, pair<int,int> > &controlPointSpace = localControlPointManagerProxy.ckLocalBranch()->controlPointSpace;

  int result;

  // Use best configuration after a certain point
  if(valueShouldBeProvidedByOptimizer()){
    result = valueProvidedByOptimizer(name);
  } 
  else {
    result = lb + randInt(ub-lb+1, name, phase_id);
    CkPrintf("Control Point \"%s\" for phase %d chosen randomly to be: %d\n", name, phase_id, result); 
  } 
   
  CkAssert(isInRange(result,ub,lb));
  thisPhaseData.controlPoints.insert(std::make_pair(string(name),result)); 
  controlPointSpace.insert(std::make_pair(string(name),std::make_pair(lb,ub))); 
  //  CkPrintf("Inserting control point value to thisPhaseData.controlPoints with value %d; thisPhaseData.controlPoints.size=%d\n", result, thisPhaseData.controlPoints.size());
  return result;
}


int controlPoint(const char *name, std::vector<int>& values){
  instrumentedPhase &thisPhaseData = localControlPointManagerProxy.ckLocalBranch()->thisPhaseData;
  const int phase_id = localControlPointManagerProxy.ckLocalBranch()->phase_id;

  int result;
  if(valueShouldBeProvidedByOptimizer()){
    result = valueProvidedByOptimizer(name);
  } 
  else { 
    result = values[randInt(values.size(), name, phase_id)];
  }

  bool found = false;
  for(int i=0;i<values.size();i++){
    if(values[i] == result)
      found = true;
  }
  CkAssert(found);

  thisPhaseData.controlPoints.insert(std::make_pair(string(name),result)); 
  return result;
}




// The index in the global array for my top row  
int redistributor2D::top_data_idx(){ 
  return (data_height * thisIndex.y) / y_chares; 
} 
 
int redistributor2D::bottom_data_idx(){ 
  return ((data_height * (thisIndex.y+1)) / y_chares) - 1; 
} 
 
int redistributor2D::left_data_idx(){ 
  return (data_width * thisIndex.x) / x_chares; 
} 
 
int redistributor2D::right_data_idx(){ 
  return ((data_width * (thisIndex.x+1)) / x_chares) - 1; 
} 
 
int redistributor2D::top_neighbor(){ 
  return (thisIndex.y + y_chares - 1) % y_chares; 
}  
   
int redistributor2D::bottom_neighbor(){ 
  return (thisIndex.y + 1) % y_chares; 
} 
   
int redistributor2D::left_neighbor(){ 
  return (thisIndex.x + x_chares - 1) % x_chares; 
} 
 
int redistributor2D::right_neighbor(){ 
  return (thisIndex.x + 1) % x_chares; 
} 
  
  
/// the width of the non-ghost part of the local partition 
int redistributor2D::mywidth(){ 
  return right_data_idx() - left_data_idx() + 1; 
} 
   
   
// the height of the non-ghost part of the local partition 
int redistributor2D::myheight(){ 
  return bottom_data_idx() - top_data_idx() + 1; 
} 








#include "ControlPoints.def.h"

#include <charm++.h>
#include <cmath>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <vector>
#include <utility>
#include <limits>
//#include <sys/time.h>
#include <float.h>

#include "ControlPoints.decl.h"
#include "trace-controlPoints.h"
#include "LBDatabase.h"
#include "controlPoints.h"


/**
 *  \addtogroup ControlPointFramework
 *   @{
 *
 */




using namespace std;

#define CONTROL_POINT_SAMPLE_PERIOD 4000
#define NUM_SAMPLES_BEFORE_TRANSISTION 5
#define OPTIMIZER_TRANSITION 5

#define WRITEDATAFILE 1



static void periodicProcessControlPoints(void* ptr, double currWallTime);


// A pointer to this PE's controlpoint manager Proxy
/* readonly */ CProxy_controlPointManager localControlPointManagerProxy;
/* readonly */ int random_seed;


/// A reduction type that combines idle time statistics (min/max/avg etc.)
CkReduction::reducerType idleTimeReductionType;
/// A reducer that combines idle time statistics (min/max/avg etc.)
CkReductionMsg *idleTimeReduction(int nMsg,CkReductionMsg **msgs){
  double ret[3];
  if(nMsg > 0){
    CkAssert(msgs[0]->getSize()==3*sizeof(double));
    double *m=(double *)msgs[0]->getData();
    ret[0]=m[0];
    ret[1]=m[1];
    ret[2]=m[2];
  }
  for (int i=1;i<nMsg;i++) {
    CkAssert(msgs[i]->getSize()==3*sizeof(double));
    double *m=(double *)msgs[i]->getData();
    ret[0]=min(ret[0],m[0]);
    ret[1]+=m[1];
    ret[2]=max(ret[2],m[2]);
  }  
  return CkReductionMsg::buildNew(3*sizeof(double),ret);   
}
/// An initcall that registers the idle time reducer idleTimeReduction()
/*initcall*/ void registerIdleTimeReduction(void) {
  idleTimeReductionType=CkReduction::addReducer(idleTimeReduction);
}



/// A container that stores idle time statistics (min/max/avg etc.)
class idleTimeContainer {
public:
  double min;
  double avg;
  double max;
  
  idleTimeContainer(){
    min = -1.0;
    max = -1.0;
    avg = -1.0;
  }
  
  bool isValid() const{
    return (min >= 0.0 && avg >= min && max >= avg && max <= 1.0);
  }
  
  void print() const{
    if(isValid())
      CkPrintf("[%d] Idle Time is Min=%.2lf%% Avg=%.2lf%% Max=%.2lf%%\n", CkMyPe(), min*100.0, avg*100.0, max*100.0);    
    else
      CkPrintf("[%d] Idle Time is invalid\n", CkMyPe(), min*100.0, avg*100.0, max*100.0);
  }
  
}; 



/// Stores data for a phase (a time range in which a single set of control point values is used).
/// The data stored includes the control point values, a set of timings registered by the application, 
/// The critical paths detected, the max memory usage, and the idle time.
class instrumentedPhase {
public:
  std::map<string,int> controlPoints; // The control point values for this phase(don't vary within the phase)
  std::vector<double> times;  // A list of times observed for iterations in this phase

  std::vector<PathHistory> criticalPaths;

  
  int memoryUsageMB;

  idleTimeContainer idleTime;

  instrumentedPhase(){
    memoryUsageMB = -1;
  }
  
  void clear(){
    controlPoints.clear();
    times.clear();
    criticalPaths.clear();
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

  
  int medianCriticalPathIdx() const{
    // Bubble sort the critical path indices by Time
    int numPaths = criticalPaths.size();
    if(numPaths>0){
      int *sortedPaths = new int[numPaths];
      for(int i=0;i<numPaths;i++){
	sortedPaths[i] = i;
      }
      
      for(int j=0;j<numPaths;j++){
	for(int i=0;i<numPaths-1;i++){
	  if(criticalPaths[sortedPaths[i]].getTotalTime() < criticalPaths[sortedPaths[i+1]].getTotalTime()){
	    // swap sortedPaths[i], sortedPaths[i+1]
	    int tmp = sortedPaths[i+1];
	    sortedPaths[i+1] = sortedPaths[i];
	    sortedPaths[i] = tmp;
	  }
	}
      }
      int result = sortedPaths[numPaths/2];
      delete[] sortedPaths;
      return result;
    } else {
      return 0;
    }
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


/// Stores and manipulate all known instrumented phases. One instance of this exists on each PE in its local controlPointManager
class instrumentedData {
public:

  /// Stores all known instrumented phases(loaded from file, or from this run)
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
	std::vector<double>::iterator titer;
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
#if OLDMAXDOUBLE
    double best_phase_avgtime = std::numeric_limits<double>::max();
#else
    double best_phase_avgtime = DBL_MAX;
#endif

    int valid_phase_count = 0;

    std::vector<instrumentedPhase>::iterator iter;
    for(iter = phases.begin();iter!=phases.end();iter++){
      if(iter->hasValidControlPointValues()){
	valid_phase_count++;

	double total_for_phase = 0.0;
	int phase_count = 0;

	// iterate over all times for this control point configuration
	std::vector<double>::iterator titer;
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
	std::vector<double>::iterator titer;
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


/// A chare group that contains most of the control point framework data and code.
class controlPointManager : public CBase_controlPointManager {
public:
  
  char * dataFilename;
  
  instrumentedData allData;
  instrumentedPhase thisPhaseData;
  instrumentedPhase best_phase;
  
  /// The lower and upper bounds for each named control point
  std::map<string, pair<int,int> > controlPointSpace;

  /// A set of named control points whose values cannot change within a single run of an application
  std::set<string> staticControlPoints;

  /// Sets of entry point ids that are affected by some named control points
  std::map<string, std::set<int> > affectsPrioritiesEP;
  /// Sets of entry array ids that are affected by some named control points
  std::map<string, std::set<int> > affectsPrioritiesArray;

  
  /// The control points to be used in the next phase. In gotoNextPhase(), these will be used
  std::map<string,int> newControlPoints;
  /// Whether to use newControlPoints in gotoNextPhase()
  bool newControlPointsAvailable;
  
  /// A user supplied callback to call when control point values are to be changed
  CkCallback granularityCallback;
  bool haveGranularityCallback;
  bool frameworkShouldAdvancePhase;
  
  int phase_id;

  bool alreadyRequestedMemoryUsage;
  bool alreadyRequestedIdleTime;


#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  /// The path history
  PathHistory maxTerminalPathHistory;
#endif

  controlPointManager(){
    newControlPointsAvailable = false;
    alreadyRequestedMemoryUsage = false;   
    alreadyRequestedIdleTime = false;
    
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
      CcdCallFnAfterOnPE((CcdVoidFn)periodicProcessControlPoints, (void*)NULL, CONTROL_POINT_SAMPLE_PERIOD, CkMyPe());

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
	ips.times.push_back(time);
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
#if WRITEDATAFILE
    CkPrintf("============= writeDataFile() ============\n");
    ofstream outfile(dataFilename);
    allData.phases.push_back(thisPhaseData);
    allData.cleanupNames();
    outfile << allData.toString();
    outfile.close();
#else
    CkPrintf("NOT WRITING OUTPUT FILE\n");
#endif
  }

  /// User can register a callback that is called when application should advance to next phase
  void setGranularityCallback(CkCallback cb, bool _frameworkShouldAdvancePhase){
    frameworkShouldAdvancePhase = _frameworkShouldAdvancePhase;
    granularityCallback = cb;
    haveGranularityCallback = true;
  }

  /// Called periodically by the runtime to handle the control points
  /// Currently called on each PE
  void processControlPoints(){

    CkPrintf("[%d] processControlPoints() haveGranularityCallback=%d frameworkShouldAdvancePhase=%d\n", CkMyPe(), (int)haveGranularityCallback, (int)frameworkShouldAdvancePhase);

        
    if(CkMyPe() == 0 && !alreadyRequestedMemoryUsage){
      alreadyRequestedMemoryUsage = true;
      CkCallback *cb = new CkCallback(CkIndex_controlPointManager::gatherMemoryUsage(NULL), 0, thisProxy);
      // thisProxy.requestMemoryUsage(*cb);
      delete cb;
    }
    
    if(CkMyPe() == 0 && !alreadyRequestedIdleTime){
      alreadyRequestedIdleTime = true;
      CkCallback *cb = new CkCallback(CkIndex_controlPointManager::gatherIdleTime(NULL), 0, thisProxy);
      thisProxy.requestIdleTime(*cb);
      delete cb;
    }
    

    //==========================================================================================
    // Print the data for each phase
    const int s = allData.phases.size();
    CkPrintf("\n\nExamining critical paths and priorities and idle times (num phases=%d)\n", s );
    for(int p=0;p<s;++p){
      const instrumentedPhase &phase = allData.phases[p];
      const idleTimeContainer &idle = phase.idleTime;
      vector<PathHistory> const &criticalPaths = phase.criticalPaths;
      vector<double> const &times = phase.times;

      CkPrintf("Phase %d:\n", p);
      idle.print();
      CkPrintf("critical paths: (* affected by control point)\n");
      for(int i=0;i<criticalPaths.size(); i++){
	// If affected by a control point
	//	criticalPaths[i].print();


	CkPrintf("Critical Path Time=%lf : ", (double)criticalPaths[i].getTotalTime());
	for(int e=0;e<numEpIdxs;e++){

	  if(criticalPaths[i].getEpIdxCount(e)>0){
	    if(controlPointAffectsThisEP(e))
	      CkPrintf("* ");
	    CkPrintf("EP %d count=%d : ", e, (int)criticalPaths[i].getEpIdxCount(e));
	  }
	}
	for(int a=0;a<numArrayIds;a++){
	  if(criticalPaths[i].getArrayIdxCount(a)>0){
	    if(controlPointAffectsThisArray(a))
	      CkPrintf("* ");
	    CkPrintf("Array %d count=%d : ", a, (int)criticalPaths[i].getArrayIdxCount(a));
	  }
	}
	CkPrintf("\n");
	

      }
      CkPrintf("Timings:\n");
      for(int i=0;i<times.size(); i++){
	CkPrintf("%lf ", times[i]);
      }
      CkPrintf("\n");

    }
    
    CkPrintf("\n\n");






    //==========================================================================================
    // If this is a phase during which we try to adapt control point values based on critical path

    if( s%5 == 4){

      // Find the most recent phase with valid critical path data and idle time measurements      
      int whichPhase=-1;
      for(int p=0; p<s; ++p){
	const instrumentedPhase &phase = allData.phases[p];
	const idleTimeContainer &idle = phase.idleTime;
	if(idle.isValid() && phase.criticalPaths.size()>0 ){
	  whichPhase = p;
	}
      }
      
      
      CkPrintf("Examining phase %d which has a valid idle time and critical paths\n", whichPhase);
      const instrumentedPhase &phase = allData.phases[whichPhase];
      const idleTimeContainer &idle = phase.idleTime;
      
      if(idle.min > 0.1){
	CkPrintf("Min PE idle is HIGH. %.2lf%% > 10%%\n", idle.min*100.0);
	
	// Determine the median critical path for this phase
	int medianCriticalPathIdx = phase.medianCriticalPathIdx();
	const PathHistory &path = phase.criticalPaths[medianCriticalPathIdx];
	
	CkPrintf("Median Critical Path has time %lf\n", path.getTotalTime());
	
	if(phase.times[medianCriticalPathIdx] > 1.2 * path.getTotalTime()){
	  CkPrintf("The application step(%lf) is taking significantly longer than the critical path(%lf). BAD\n",phase.times[medianCriticalPathIdx], path.getTotalTime() );


	  CkPrintf("Finding control points related to the critical path\n");
	  int cpcount = 0;
	  std::set<string> controlPointsAffectingCriticalPath;


	  for(int e=0;e<numEpIdxs;e++){
	    if(path.getEpIdxCount(e)>0){
	      
	      std::map<string, std::set<int> >::iterator iter;
	      for(iter=affectsPrioritiesEP.begin(); iter!= affectsPrioritiesEP.end(); ++iter){
		if(iter->second.count(e)>0){
		  controlPointsAffectingCriticalPath.insert(iter->first);
		  CkPrintf("Control Point \"%s\" affects the critical path\n", iter->first.c_str());
		  cpcount++;
		}
	      }
	      
	    }
	  }
	  

	  if(cpcount==0){
	    CkPrintf("No control points are known to affect the critical path\n");
	  } else {
	    double beginTime = CmiWallTimer();

	    CkPrintf("Attempting to modify control point values for %d control points that affect the critical path\n", controlPointsAffectingCriticalPath.size());
	    
	    newControlPoints = phase.controlPoints;
	    
	    if(frameworkShouldAdvancePhase){
	      gotoNextPhase();	
	    }
	    
	    if(haveGranularityCallback){ 
	      controlPointMsg *msg = new(0) controlPointMsg;
	      granularityCallback.send(msg);
	    }
	    
	    
	    // adjust the control points that can affect the critical path

	    char textDescription[4096*2];
	    textDescription[0] = '\0';

	    std::map<string,int>::iterator newCP;
	    for(newCP = newControlPoints.begin(); newCP != newControlPoints.end(); ++ newCP){
	      if( controlPointsAffectingCriticalPath.count(newCP->first) > 0 ){
		// decrease the value (increase priority) if within range
		int lowerbound = controlPointSpace[newCP->first].first;
		if(newCP->second > lowerbound){
		  newControlPointsAvailable = true;
  		  newCP->second --;
		}
	      }
	    }
	   
	    // Create a string for a projections user event
	    if(1){
	      std::map<string,int>::iterator newCP;
	      for(newCP = newControlPoints.begin(); newCP != newControlPoints.end(); ++ newCP){
		sprintf(textDescription+strlen(textDescription), "<br>\"%s\"=%d", newCP->first.c_str(), newCP->second);
	      }
	    }



	    char *userEventString = new char[4096*2];
	    sprintf(userEventString, "Adjusting control points affecting critical path: %s ***", textDescription);
	    
	    static int userEventCounter = 20000;
	    CkPrintf("USER EVENT: %s\n", userEventString);
	    
	    traceRegisterUserEvent(userEventString, userEventCounter); 
	    traceUserBracketEvent(userEventCounter, beginTime, CmiWallTimer());
	    userEventCounter++;
	    
	  }
	  
	} else {
	  CkPrintf("The application step(%lf) is dominated mostly by the critical path(%lf). GOOD\n",phase.times[medianCriticalPathIdx], path.getTotalTime() );
	}
	
	
      }
      
    }
    
    CkPrintf("\n");
    
    
    
    if(haveGranularityCallback){
      if(frameworkShouldAdvancePhase){
	gotoNextPhase();	
      }
      
      controlPointMsg *msg = new(0) controlPointMsg;
      granularityCallback.send(msg); 
    }
    
    
    
  }
  
  /// Determine if any control point is known to affect an entry method
  bool controlPointAffectsThisEP(int ep){
    std::map<string, std::set<int> >::iterator iter;
    for(iter=affectsPrioritiesEP.begin(); iter!= affectsPrioritiesEP.end(); ++iter){
      if(iter->second.count(ep)>0){
	return true;
      }
    }
    return false;    
  }
  
  /// Determine if any control point is known to affect a chare array  
  bool controlPointAffectsThisArray(int array){
    std::map<string, std::set<int> >::iterator iter;
    for(iter=affectsPrioritiesArray.begin(); iter!= affectsPrioritiesArray.end(); ++iter){
      if(iter->second.count(array)>0){
	return true;
      }
    }
    return false;   
  }
  
  /// The data from the previous phase
  instrumentedPhase *previousPhaseData(){
    int s = allData.phases.size();
    if(s >= 1 && phase_id > 0) {
      return &(allData.phases[s-1]);
    } else {
      return NULL;
    }
  }
  

  /// Called by either the application or the Control Point Framework to advance to the next phase  
  void gotoNextPhase(){

    LBDatabase * myLBdatabase = LBDatabaseObj();

#if CMK_LBDB_ON
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
    
#endif    
    
    
    // increment phase id
    phase_id++;
    
    CkPrintf("Now in phase %d\n", phase_id);
    
    // save a copy of the timing information from this phase
    allData.phases.push_back(thisPhaseData);
    
    // clear the timing information that will be used for the next phase
    thisPhaseData.clear();
    
  }

  /// An application uses this to register an instrumented timing for this phase
  void setTiming(double time){
    thisPhaseData.times.push_back(time);
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
       
    // First we should register this currently executing message as a path, because it is likely an important one to consider.
    registerTerminalEntryMethod();
    
    // save the critical path for this phase
    thisPhaseData.criticalPaths.push_back(maxTerminalPathHistory);
    maxTerminalPathHistory.reset();
    
    
    // Reset the counts for the currently executing message
    resetThisEntryPath();
    
    
#endif
    
  }
  
  /// Entry method called on all PEs to request memory usage
  void requestIdleTime(CkCallback cb){
    double i = localControlPointTracingInstance()->idleRatio();
    double idle[3];
    idle[0] = i;
    idle[1] = i;
    idle[2] = i;
    
    localControlPointTracingInstance()->resetTimings();

    contribute(3*sizeof(double),idle,idleTimeReductionType, cb);
  }
  
  /// All processors reduce their memory usages in requestIdleTime() to this method
  void gatherIdleTime(CkReductionMsg *msg){
    int size=msg->getSize() / sizeof(double);
    CkAssert(size==3);
    double *r=(double *) msg->getData();
        
    instrumentedPhase* prevPhase = previousPhaseData();
    if(prevPhase != NULL){
      CkPrintf("Storing idle time measurements\n");
      prevPhase->idleTime.min = r[0];
      prevPhase->idleTime.avg = r[1]/CkNumPes();
      prevPhase->idleTime.max = r[2];
      prevPhase->idleTime.print();
      CkPrintf("Stored idle time min=%lf in prevPhase=%p\n", prevPhase->idleTime.min, prevPhase);
    } else {
      CkPrintf("No place to store idle time measurements\n");
    }
    
    alreadyRequestedIdleTime = false;
    delete msg;
  }
  


  /// Entry method called on all PEs to request memory usage
  void requestMemoryUsage(CkCallback cb){
    int m = CmiMaxMemoryUsage() / 1024 / 1024;
    CmiResetMaxMemory();
    CkPrintf("PE %d Memory Usage is %d MB\n",CkMyPe(), m);
    contribute(sizeof(int),&m,CkReduction::max_int, cb);
  }

  /// All processors reduce their memory usages to this method
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


  /// An entry method used to both register and reduce the maximal critical paths back to PE 0
  void registerTerminalPath(PathHistory &path){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
  
    double beginTime = CmiWallTimer();
        
    if(maxTerminalPathHistory.updateMax(path) ) {
      // The new path is more critical than the previous one
      // propogate it towards processor 0 in a binary tree
      if(CkMyPe() > 0){
	int dest = (CkMyPe() -1) / 2;
	//	CkPrintf("Forwarding better critical path from PE %d to %d\n", CkMyPe(), dest);

	// This is part of a reduction-like propagation of the maximum back to PE 0
	resetThisEntryPath();

	thisProxy[dest].registerTerminalPath(path);
      }
    }

    traceRegisterUserEvent("registerTerminalPath", 100); 
    traceUserBracketEvent(100, beginTime, CmiWallTimer());
#endif
  }

  /// Print the maximal known critical path on this PE
  void printTerminalPath(){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
    maxTerminalPathHistory.print();
#endif
  }

  /// Reset the maximal known critical path on this PE  
  void resetTerminalPath(){
#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
    maxTerminalPathHistory.reset();
#endif
  }


  /// Inform the control point framework that a named control point affects the priorities of some array  
  void associatePriorityArray(const char *name, int groupIdx){
    CkPrintf("Associating control point \"%s\" affects priority of array id=%d\n", name, groupIdx );
    
    if(affectsPrioritiesArray.count(std::string(name)) > 0 ) {
      affectsPrioritiesArray[std::string(name)].insert(groupIdx);
    } else {
      std::set<int> s;
      s.insert(groupIdx);
      affectsPrioritiesArray[std::string(name)] = s;
    }
    
#if DEBUG   
    std::map<string, std::set<int> >::iterator f;
    for(f=affectsPrioritiesArray.begin(); f!=affectsPrioritiesArray.end();++f){
      std::string name = f->first;
      std::set<int> &vals = f->second;
      cout << "Control point " << name << " affects arrays: ";
      std::set<int>::iterator i;
      for(i=vals.begin(); i!=vals.end();++i){
	cout << *i << " ";
      }
      cout << endl;
    }
#endif
    
  }
  
  /// Inform the control point framework that a named control point affects the priority of some entry method
  void associatePriorityEntry(const char *name, int idx){
    CkPrintf("Associating control point \"%s\" with EP id=%d\n", name, idx);

      if(affectsPrioritiesEP.count(std::string(name)) > 0 ) {
      affectsPrioritiesEP[std::string(name)].insert(idx);
    } else {
      std::set<int> s;
      s.insert(idx);
      affectsPrioritiesEP[std::string(name)] = s;
    }
    
#if DEBUG
    std::map<string, std::set<int> >::iterator f;
    for(f=affectsPrioritiesEP.begin(); f!=affectsPrioritiesEP.end();++f){
      std::string name = f->first;
      std::set<int> &vals = f->second;
      cout << "Control point " << name << " affects EP: ";
      std::set<int>::iterator i;
      for(i=vals.begin(); i!=vals.end();++i){
	cout << *i << " ";
      }
      cout << endl;
    }
#endif


  }
  
};


/// An interface callable by the application.
void gotoNextPhase(){
  localControlPointManagerProxy.ckLocalBranch()->gotoNextPhase();
}


/// A mainchare that is used just to create our controlPointManager group at startup
class controlPointMain : public CBase_controlPointMain {
public:
  controlPointMain(CkArgMsg* args){
#if OLDRANDSEED
    struct timeval tp;
    gettimeofday(& tp, NULL);
    random_seed = (int)tp.tv_usec ^ (int)tp.tv_sec;
#else
    double time = CmiWallTimer();
    int sec = (int)time;
    int usec = (int)((time - (double)sec)*1000000.0);
    random_seed =  sec ^ usec;
#endif

    localControlPointManagerProxy = CProxy_controlPointManager::ckNew();
  }
  ~controlPointMain(){}
};

/// An interface callable by the application.
void registerGranularityChangeCallback(CkCallback cb, bool frameworkShouldAdvancePhase){
  CkAssert(CkMyPe() == 0);
  CkPrintf("registerGranularityChangeCallback\n");
  localControlPointManagerProxy.ckLocalBranch()->setGranularityCallback(cb, frameworkShouldAdvancePhase);
}

/// An interface callable by the application.
void registerControlPointTiming(double time){
  CkAssert(CkMyPe() == 0);
#if DEBUG>0
  CkPrintf("Program registering its own timing with registerControlPointTiming(time=%lf)\n", time);
#endif
  localControlPointManagerProxy.ckLocalBranch()->setTiming(time);
}

/// Shutdown the control point framework, writing data to disk if necessary
extern "C" void controlPointShutdown(){
  CkAssert(CkMyPe() == 0);
  CkPrintf("[%d] controlPointShutdown() at CkExit()\n", CkMyPe());
  localControlPointManagerProxy.ckLocalBranch()->writeDataFile();
  CkExit();
}

/// A function called at startup on each node to register controlPointShutdown() to be called at CkExit()
void controlPointInitNode(){
  CkPrintf("controlPointInitNode()\n");
  registerExitFn(controlPointShutdown);
}

/// Called periodically to allow control point framework to do things periodically
static void periodicProcessControlPoints(void* ptr, double currWallTime){
#ifdef DEBUG
  CkPrintf("[%d] periodicProcessControlPoints()\n", CkMyPe());
#endif
  localControlPointManagerProxy.ckLocalBranch()->processControlPoints();
  CcdCallFnAfterOnPE((CcdVoidFn)periodicProcessControlPoints, (void*)NULL, CONTROL_POINT_SAMPLE_PERIOD, CkMyPe());
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


/// Should an optimizer determine the control point values
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





/// Determine a control point value using some optimization scheme (use max known, simmulated annealling, 
/// user observed characteristic to adapt specific control point values.
/// @note eventually there should be a plugin system where multiple schemes can be plugged in(similar to LB)
int valueProvidedByOptimizer(const char * name){
  const int phase_id = localControlPointManagerProxy.ckLocalBranch()->phase_id;
  const int effective_phase = localControlPointManagerProxy.ckLocalBranch()->allData.phases.size();


#define OPTIMIZER_ADAPT_CRITICAL_PATHS 1
  
  // -----------------------------------------------------------
#if OPTIMIZER_ADAPT_CRITICAL_PATHS
  // This scheme will return the median value for the range 
  // early on, and then will transition over to the new control points
  // determined by the critical path adapting code
  if(localControlPointManagerProxy.ckLocalBranch()->newControlPointsAvailable){
    int result = localControlPointManagerProxy.ckLocalBranch()->newControlPoints[string(name)];
    CkPrintf("valueProvidedByOptimizer(): Control Point \"%s\" for phase %d  from \"newControlPoints\" is: %d\n", name, phase_id, result);
    return result;
  } 
  
  std::map<string, pair<int,int> > &controlPointSpace = localControlPointManagerProxy.ckLocalBranch()->controlPointSpace;  

  if(controlPointSpace.count(std::string(name))>0){
    int minValue =  controlPointSpace[std::string(name)].first;
    int maxValue =  controlPointSpace[std::string(name)].second;
    return (minValue+maxValue)/2;
  }
  
  
  // -----------------------------------------------------------
#elif OPTIMIZER_USE_BEST_TIME  
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

  // -----------------------------------------------------------
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

  // -----------------------------------------------------------
#else
  // Exhaustive search

  std::map<string, pair<int,int> > &controlPointSpace = localControlPointManagerProxy.ckLocalBranch()->controlPointSpace;
  std::set<string> &staticControlPoints = localControlPointManagerProxy.ckLocalBranch()->staticControlPoints;  
   
  int numDimensions = controlPointSpace.size();
  CkAssert(numDimensions > 0);
  
  vector<int> lowerBounds(numDimensions);
  vector<int> upperBounds(numDimensions); 
  
  int d=0;
  std::map<string, pair<int,int> >::iterator iter;
  for(iter = controlPointSpace.begin(); iter != controlPointSpace.end(); iter++){
    //    CkPrintf("Examining dimension %d\n", d);

#if DEBUG
    string name = iter->first;
    if(staticControlPoints.count(name) >0 ){
      cout << " control point " << name << " is static " << endl;
    } else{
      cout << " control point " << name << " is not static " << endl;
    }
#endif

    lowerBounds[d] = iter->second.first;
    upperBounds[d] = iter->second.second;
    d++;
  }
   

  vector<std::string> s(numDimensions);
  d=0;
  for(std::map<string, pair<int,int> >::iterator niter=controlPointSpace.begin(); niter!=controlPointSpace.end(); niter++){
    s[d] = niter->first;
    // cout << "s[" << d << "]=" << s[d] << endl;
    d++;
  }
  
  
  // Create the first possible configuration
  vector<int> config = lowerBounds;
  config.push_back(0);
  
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

    int l1 = CmiLog2(fine_granularity);
    int l2 = CmiLog2(coarse_granularity);
  
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


/// Get control point value from range of integers [lb,ub]
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

/// Get control point value from set of provided integers
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




/// Inform the control point framework that a named control point affects the priorities of some array  
void controlPointPriorityArray(const char *name, CProxy_ArrayBase &arraybase){
  CkGroupID aid = arraybase.ckGetArrayID();
  int groupIdx = aid.idx;
  localControlPointManagerProxy.ckLocalBranch()->associatePriorityArray(name, groupIdx);
  //  CkPrintf("Associating control point \"%s\" with array id=%d\n", name, groupIdx );
}


/// Inform the control point framework that a named control point affects the priorities of some entry method  
void controlPointPriorityEntry(const char *name, int idx){
  localControlPointManagerProxy.ckLocalBranch()->associatePriorityEntry(name, idx);
  //  CkPrintf("Associating control point \"%s\" with EP id=%d\n", name, idx);
}






/// The index in the global array for my top row  
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
   
   
/// the height of the non-ghost part of the local partition 
int redistributor2D::myheight(){ 
  return bottom_data_idx() - top_data_idx() + 1; 
} 












#ifdef USE_CRITICAL_PATH_HEADER_ARRAY


/// Inform control point framework that the just executed entry method is a terminal one. This is used to maintain the maximum known critical path .
void registerTerminalEntryMethod(){
  localControlPointManagerProxy.ckLocalBranch()->registerTerminalPath(currentlyExecutingMsg->pathHistory);
}

void printPECriticalPath(){
  localControlPointManagerProxy.ckLocalBranch()->printTerminalPath();
}

void resetPECriticalPath(){
  localControlPointManagerProxy.ckLocalBranch()->resetTerminalPath();
}

#endif


/*! @} */


#include "ControlPoints.def.h"

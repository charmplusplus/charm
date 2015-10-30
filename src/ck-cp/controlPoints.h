/** 

    A system for exposing application and runtime "control points" 
    to the dynamic optimization framework.

*/


#ifndef __CONTROLPOINTS_H__
#define __CONTROLPOINTS_H__

#include "conv-config.h"

#if CMK_WITH_CONTROLPOINT


#include "ControlPoints.decl.h"

#include <vector>
#include <map>
#include <cmath>
#include <pup_stl.h>
#include <string>
#include <set>
#include <cmath>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <float.h>
#include <charm-api.h>

#include "LBDatabase.h"
#include "arrayRedistributor.h"
#include "pathHistory.h" 


#include <cp_effects.h>

/**
 * \addtogroup ControlPointFramework
 *   @{
 */

#define DEBUG 0

/* readonly */ extern CProxy_controlPointManager controlPointManagerProxy;
/* readonly */ extern int random_seed;
/* readonly */ extern long controlPointSamplePeriod;
/* readonly */ extern int whichTuningScheme;
/* readonly */ extern bool writeDataFileAtShutdown;
/* readonly */ extern bool shouldFilterOutputData;
/* readonly */ extern bool loadDataFileAtStartup;
/* readonly */ extern char CPDataFilename[512];



void registerCPChangeCallback(CkCallback cb, bool frameworkShouldAdvancePhase);

void setFrameworkAdvancePhase(bool frameworkShouldAdvancePhase);


void registerControlPointTiming(double time);

/// Called once each application step. Can be used instead of registerControlPointTiming()
void controlPointTimingStamp();
FDECL void FTN_NAME(CONTROLPOINTTIMINGSTAMP,controlpointtimingstamp)();


/// The application specifies that it is ready to proceed to a new set of control point values.
/// This should be called after registerControlPointTiming()
/// This should be called before calling controlPoint()
void gotoNextPhase();
FDECL void FTN_NAME(GOTONEXTPHASE,gotonextphase)();

/// Return an integral power of 2 between c1 and c2
/// The value returned will likely change between subsequent invocations
int controlPoint2Pow(const char *name, int c1, int c2);

/// Return an integer between lb and ub inclusive
/// The value returned will likely change between subsequent invocations
int controlPoint(const char *name, int lb, int ub);

/// A fortran callable one. I couldn't figure out how to pass a string from fortran to C++ yet
/// So far fortran can only have one control point
FDECL int FTN_NAME(CONTROLPOINT,controlpoint)(CMK_TYPEDEF_INT4 *lb, CMK_TYPEDEF_INT4 *ub);


/// Return an integer from the provided vector of values
/// The value returned will likely change between subsequent invocations
int controlPoint(const char *name, std::vector<int>& values);

/// Write output data to disk. Callable from user program (for example, to periodically flush to disk if program might run out of time, or NAMD)
void ControlPointWriteOutputToDisk();

/// The application specifies that it is ready to proceed to a new set of control point values.
/// This should be called after registerControlPointTiming()
/// This should be called before calling controlPoint()
void gotoNextPhase();




/// A message used for signaling changes in control point values
class controlPointMsg : public CMessage_controlPointMsg {
 public:
  char *data;
};





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


/// A container that stores overhead statistics (min/max/avg etc.)
class overheadContainer {
public:
  double min;
  double avg;
  double max;
  
  overheadContainer(){
    min = -1.0;
    max = -1.0;
    avg = -1.0;
  }
  
  bool isValid() const{
    return (min >= 0.0 && avg >= min && max >= avg && max <= 1.0);
  }
  
  void print() const{
    if(isValid())
      CkPrintf("[%d] Overhead Time is Min=%.2lf%% Avg=%.2lf%% Max=%.2lf%%\n", CkMyPe(), min*100.0, avg*100.0, max*100.0);    
    else
      CkPrintf("[%d] Overhead Time is invalid\n", CkMyPe(), min*100.0, avg*100.0, max*100.0);
  }
  
}; 



/// Stores data for a phase (a time range in which a single set of control point values is used).
/// The data stored includes the control point values, a set of timings registered by the application, 
/// The critical paths detected, the max memory usage, and the idle time.
class instrumentedPhase {
public:
  std::map<std::string,int> controlPoints; // The control point values for this phase(don't vary within the phase)
  std::vector<double> times;  // A list of times observed for iterations in this phase

#if USE_CRITICAL_PATH_HEADER_ARRAY
  std::vector<PathHistoryTableEntry> criticalPaths;
#endif
  
  double memoryUsageMB;

  idleTimeContainer idleTime;
  overheadContainer overheadTime;


  /** Approximately records the average message size for an entry method. */
  double bytesPerInvoke;

  /** Records the average grain size (might be off a bit due to non application entry methods), in seconds */
  double grainSize;


  instrumentedPhase(){
    memoryUsageMB = -1.0;
    grainSize = -1.0;
    bytesPerInvoke = -1.0;
  }
  
  void clear(){
    controlPoints.clear();
    times.clear();
    memoryUsageMB = -1.0;
    grainSize = -1.0;
    bytesPerInvoke = -1.0;
    //    criticalPaths.clear();
  }

  // Provide a previously computed value, or a value from a previous run
  bool haveValueForName(const char* name){
    std::string n(name);
    return (controlPoints.count(n)>0);
  }

  void operator=(const instrumentedPhase& p){
    controlPoints = p.controlPoints;
    times = p.times;
    memoryUsageMB = p.memoryUsageMB;
  }



  bool operator<(const instrumentedPhase& p){
    CkAssert(hasSameKeysAs(p)); 
    std::map<std::string,int>::iterator iter1 = controlPoints.begin();
    std::map<std::string,int>::const_iterator iter2 = p.controlPoints.begin();
    for(;iter1 != controlPoints.end() && iter2 != p.controlPoints.end(); iter1++, iter2++){
      if(iter1->second < iter2->second){
	return true;
      }
    }
    return false;
  }


  // Determines if the control point values and other information exists
  bool hasValidControlPointValues(){
    std::map<std::string,int>::iterator iter;
    for(iter = controlPoints.begin();iter != controlPoints.end(); iter++){
      if(iter->second == -1){ 
        return false; 
      }  
    }
    return true;
  }

  
//   int medianCriticalPathIdx() const{
//     // Bubble sort the critical path indices by Time
//     int numPaths = criticalPaths.size();
//     if(numPaths>0){
//       int *sortedPaths = new int[numPaths];
//       for(int i=0;i<numPaths;i++){
// 	sortedPaths[i] = i;
//       }
      
//       for(int j=0;j<numPaths;j++){
// 	for(int i=0;i<numPaths-1;i++){
// 	  if(criticalPaths[sortedPaths[i]].getTotalTime() < criticalPaths[sortedPaths[i+1]].getTotalTime()){
// 	    // swap sortedPaths[i], sortedPaths[i+1]
// 	    int tmp = sortedPaths[i+1];
// 	    sortedPaths[i+1] = sortedPaths[i];
// 	    sortedPaths[i] = tmp;
// 	  }
// 	}
//       }
//       int result = sortedPaths[numPaths/2];
//       delete[] sortedPaths;
//       return result;
//     } else {
//       return 0;
//     }
//   }



  bool operator==(const instrumentedPhase& p){
    CkAssert(hasSameKeysAs(p));
    std::map<std::string,int>::iterator iter1 = controlPoints.begin();
    std::map<std::string,int>::const_iterator iter2 = p.controlPoints.begin();
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

    std::map<std::string,int>::iterator iter1 = controlPoints.begin(); 
    std::map<std::string,int>::const_iterator iter2 = p.controlPoints.begin(); 

    for(;iter1 != controlPoints.end() && iter2 != p.controlPoints.end(); iter1++, iter2++){  
      if(iter1->first != iter2->first)
	return false;
    } 

    return true; 
  }


  void addAllNames(std::set<std::string> names_) {
    
    std::set<std::string> names = names_;
    
    // Remove all the names that we already have
    std::map<std::string,int>::iterator iter;
    
    for(iter = controlPoints.begin(); iter != controlPoints.end(); iter++){
      names.erase(iter->first);
    }
    
    // Add -1 values for each name we didn't find
    std::set<std::string>::iterator iter2;
    for(iter2 = names.begin(); iter2 != names.end(); iter2++){
      controlPoints.insert(std::make_pair(*iter2,-1));
      CkPrintf("One of the datasets was missing a value for %s, so -1 was used\n", iter2->c_str());
    }

  }
  
  
  
  void print() const {
    std::map<std::string,int>::const_iterator iter;

    if(controlPoints.size() == 0){
      CkPrintf("no control point values found\n");
    }
    
    for(iter = controlPoints.begin(); iter != controlPoints.end(); iter++){
      const std::string name = iter->first;
      const int val = iter->second;
      CkPrintf("%s ---> %d\n",  name.c_str(),  val);
    } 
    
  }

  /** Determine the median time for this phase */
  double medianTime(){
    std::vector<double> sortedTimes = times;
    std::sort(sortedTimes.begin(), sortedTimes.end());
    if(sortedTimes.size()>0){
    	return sortedTimes[sortedTimes.size() / 2];
    } else {
    	CkAbort("Cannot compute medianTime for empty sortedTimes vector");
    	return -1;
    }
  }

  
};


/// Stores and manipulate all known instrumented phases. One instance of this exists on each PE in its local controlPointManager
class instrumentedData {
public:

  /// Stores all known instrumented phases(loaded from file, or from this run)
  std::vector<instrumentedPhase*> phases;

  /// get control point names for all phases
  std::set<std::string> getNames(){
    std::set<std::string> names;
    
    std::vector<instrumentedPhase*>::iterator iter;
    for(iter = phases.begin();iter!=phases.end();iter++) {
      
      std::map<std::string,int>::iterator iter2;
      for(iter2 = (*iter)->controlPoints.begin(); iter2 != (*iter)->controlPoints.end(); iter2++){
	names.insert(iter2->first);
      }
      
    }  
    return names;

  } 


  void cleanupNames(){
    std::set<std::string> names = getNames();
    
    std::vector<instrumentedPhase*>::iterator iter;
    for(iter = phases.begin();iter!=phases.end();iter++) {
      (*iter)->addAllNames(names);
    }
  }


  /// Remove one phase with invalid control point values if found
  bool filterOutOnePhase(){
#if 1
    // Note: calling erase on a vector will invalidate any iterators beyond the deletion point
    std::vector<instrumentedPhase*>::iterator iter;
    for(iter = phases.begin(); iter != phases.end(); iter++) {
      if(! (*iter)->hasValidControlPointValues()  || (*iter)->times.size()==0){
	// CkPrintf("Filtered out a phase with incomplete control point values\n");
	phases.erase(iter);
	return true;
      } else {
	//	CkPrintf("Not filtering out some phase with good control point values\n");
      }
    }
#endif
    return false;
  }
  
  /// Drop any phases that do not contain timings or control point values
  void filterOutIncompletePhases(){
    bool done = false;
    while(filterOutOnePhase()){
      // do nothing
    }
  }


  std::string toString(){
    std::ostringstream s;

    // HEADER:
    s << "# HEADER:\n";
    s << "# Data for use with Isaac Dooley's Control Point Framework\n";
    s << "# Number of instrumented timings in this file:\n"; 
    s << phases.size() << "\n" ;
    
    if(phases.size() > 0){
      
      std::map<std::string,int> &ps = phases[0]->controlPoints; 
      std::map<std::string,int>::iterator cpiter;

      // SCHEMA:
      s << "# SCHEMA:\n";
      s << "# number of named control points:\n";
      s << ps.size() << "\n";    
      for(cpiter = ps.begin(); cpiter != ps.end(); cpiter++){
	s << cpiter->first << "\n";
      }
      
      // DATA:
      s << "# DATA:\n";
      s << "# There are " << ps.size()  << " control points\n";
      s << "# number of recorded phases: " << phases.size() << "\n";
      
      s << "# Memory (MB)\tIdle Min\tIdle Avg\tIdle Max\tOverhead Min\tOverhead Avg\tOverhead Max\tByte Per Invoke\tGrain Size\t";
      for(cpiter = ps.begin(); cpiter != ps.end(); cpiter++){
	s << cpiter->first << "\t";
      }
      s << "Median Timing\tTimings\n";
      
   
      std::vector<instrumentedPhase*>::iterator runiter;
      for(runiter=phases.begin();runiter!=phases.end();runiter++){
	
	// Print the memory usage
	s << (*runiter)->memoryUsageMB << "\t"; 

	s << (*runiter)->idleTime.min << "\t" << (*runiter)->idleTime.avg << "\t" << (*runiter)->idleTime.max << "\t";
	s << (*runiter)->overheadTime.min << "\t" << (*runiter)->overheadTime.avg << "\t" << (*runiter)->overheadTime.max << "\t";


	s << (*runiter)->bytesPerInvoke << "\t";

	s << (*runiter)->grainSize << "\t";


	// Print the control point values
	for(cpiter = (*runiter)->controlPoints.begin(); cpiter != (*runiter)->controlPoints.end(); cpiter++){ 
	  s << cpiter->second << "\t"; 
	}

	// Print the median time
	if((*runiter)->times.size()>0){
		s << (*runiter)->medianTime() << "\t";
	} else {
		s << "-1\t";
	}

	// Print the times
	std::vector<double>::iterator titer;
	for(titer = (*runiter)->times.begin(); titer != (*runiter)->times.end(); titer++){
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
      instrumentedPhase *firstpoint = phases[0];
      std::vector<instrumentedPhase*>::iterator iter;
      for(iter = phases.begin();iter!=phases.end();iter++){
	CkAssert( firstpoint->hasSameKeysAs(*(*iter)));
      }  
    } 
  }


  // Find the fastest time from previous runs
  instrumentedPhase* findBest(){
    CkAssert(phases.size()>1);

    double total_time = 0.0; // total for all times
    int total_count = 0;

    instrumentedPhase * best_phase;

#if OLDMAXDOUBLE
    double best_phase_avgtime = std::numeric_limits<double>::max();
#else
    double best_phase_avgtime = DBL_MAX;
#endif

    int valid_phase_count = 0;

    std::vector<instrumentedPhase*>::iterator iter;
    for(iter = phases.begin();iter!=phases.end();iter++){
      if((*iter)->hasValidControlPointValues()){
	valid_phase_count++;

	double total_for_phase = 0.0;
	int phase_count = 0;

	// iterate over all times for this control point configuration
	std::vector<double>::iterator titer;
	for(titer = (*iter)->times.begin(); titer != (*iter)->times.end(); titer++){
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
      if((*iter)->hasValidControlPointValues()){
	std::vector<double>::iterator titer;
	for(titer = (*iter)->times.begin(); titer != (*iter)->times.end(); titer++){
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


/** A class that implements the Nelder Mead Simplex Optimization Algorithm */
class simplexScheme {
private:
        /** The states used by the Nelder Mead Simplex Algorithm. 
	    The transitions between these states are displayed in the NelderMeadStateDiagram.pdf diagram.
	*/
	typedef enum simplexStateEnumT {beginning, reflecting, expanding, contracting, doneExpanding, stillContracting}  simplexStateT;

	/// The indices into the allData->phases that correspond to the current simplex used one of the tuning schemes.
	std::set<int> simplexIndices;
	simplexStateT simplexState;

	 /// Reflection coefficient
	double alpha;
	// Contraction coefficient
	double beta;
	// Expansion coefficient
	double gamma;


	/// The first phase that was used by the this scheme. This helps us ignore a few startup phases that are out of our control.
	int firstSimplexPhase;

	// Worst performing point in the simplex
	int worstPhase;
	double worstTime;
	std::vector<double> worst; // p_h

	// Centroid of remaining points in the simplex
	std::vector<double> centroid;

	// Best performing point in the simplex
	int bestPhase;
	double bestTime;
	std::vector<double> best;

	// P*
	std::vector<double> P;
	int pPhase;

	// P**
	std::vector<double> P2;
	int p2Phase;

	// A set of phases for which (P_i+P_l)/2 ought to be evaluated
	std::set<int> stillMustContractList;

	bool useBestKnown;

	void computeCentroidBestWorst(std::map<std::string, std::pair<int,int> > & controlPointSpace, std::map<std::string,int> &newControlPoints, const int phase_id, instrumentedData &allData);

	void doReflection(std::map<std::string, std::pair<int,int> > & controlPointSpace, std::map<std::string,int> &newControlPoints, const int phase_id, instrumentedData &allData);
	void doExpansion(std::map<std::string, std::pair<int,int> > & controlPointSpace, std::map<std::string,int> &newControlPoints, const int phase_id, instrumentedData &allData);
	void doContraction(std::map<std::string, std::pair<int,int> > & controlPointSpace, std::map<std::string,int> &newControlPoints, const int phase_id, instrumentedData &allData);


	std::vector<double> pointCoords(instrumentedData &allData, int i);

		inline int keepInRange(int v, int lb, int ub){
			if(v < lb)
				return lb;
			if(v > ub)
				return ub;
			return v;
		}

		void printSimplex(instrumentedData &allData){
			char s[2048];
			s[0] = '\0';
			for(std::set<int>::iterator iter = simplexIndices.begin(); iter != simplexIndices.end(); ++iter){
				sprintf(s+strlen(s), "%d: ", *iter);

				for(std::map<std::string,int>::iterator citer = allData.phases[*iter]->controlPoints.begin(); citer != allData.phases[*iter]->controlPoints.end(); ++citer){
					sprintf(s+strlen(s), " %d", citer->second);
				}

				sprintf(s+strlen(s), "\n");
			}
			CkPrintf("Current simplex is:\n%s\n", s);
		}

public:

	simplexScheme() :
		simplexState(beginning),
		alpha(1.0), beta(0.5), gamma(2.0),
		firstSimplexPhase(-1),
		useBestKnown(false)
	{
		// Make sure the coefficients are reasonable
		CkAssert(alpha >= 0);
		CkAssert(beta >= 0.0 && beta <= 1.0);
		CkAssert(gamma >= 1.0);
	}

	void adapt(std::map<std::string, std::pair<int,int> > & controlPointSpace, std::map<std::string,int> &newControlPoints, const int phase_id, instrumentedData &allData);

};



class controlPointManager : public CBase_controlPointManager {
public:
    
  instrumentedData allData;
  
  /// The lower and upper bounds for each named control point
  std::map<std::string, std::pair<int,int> > controlPointSpace;

  /// A set of named control points whose values cannot change within a single run of an application
  std::set<std::string> staticControlPoints;

  /// @deprecated Sets of entry point ids that are affected by some named control points
  std::map<std::string, std::set<int> > affectsPrioritiesEP;
  /// @deprecated Sets of entry array ids that are affected by some named control points
  std::map<std::string, std::set<int> > affectsPrioritiesArray;

  
  /// The control points to be used in the next phase. In gotoNextPhase(), these will be used
  std::map<std::string,int> newControlPoints;
  int generatedPlanForStep;

  simplexScheme s;

  
  /// A user supplied callback to call when control point values are to be changed
  CkCallback controlPointChangeCallback;
  bool haveControlPointChangeCallback;
  bool frameworkShouldAdvancePhase;
  
  int phase_id;

  bool alreadyRequestedMemoryUsage;
  bool alreadyRequestedIdleTime;
  bool alreadyRequestedAll;

  bool exitWhenReady;


  controlPointManager();
     
  ~controlPointManager();

  void pup(PUP::er &p);
  
 controlPointManager(CkMigrateMessage* m) : CBase_controlPointManager(m) {
    // CkPrintf("Warning: control point framework likely won't work after checkpoint restart, please fix this problem if you need the functionality.");
  }


  /// Loads the previous run data file
  void loadDataFile();

  /// Add the current data to allData and output it to a file
  void writeDataFile();

  /// User can register a callback that is called when application should advance to next phase
  void setCPCallback(CkCallback cb, bool _frameworkShouldAdvancePhase);

  void setFrameworkAdvancePhase(bool _frameworkShouldAdvancePhase);

  /// Called periodically by the runtime to handle the control points
  /// Currently called on each PE
  void processControlPoints();
  
  /// Determine if any control point is known to affect an entry method
  bool controlPointAffectsThisEP(int ep);
  
  /// Determine if any control point is known to affect a chare array  
  bool controlPointAffectsThisArray(int array);

  /// Generate a plan (new control point values) once per phase
  void generatePlan();

  /// The data for the current phase
  instrumentedPhase *currentPhaseData();

  /// The data from the previous phase
  instrumentedPhase *previousPhaseData();

  /// The data from two phases back
  instrumentedPhase *twoAgoPhaseData();

  /// Called by either the application or the Control Point Framework to advance to the next phase  
  void gotoNextPhase();

  /// An application uses this to register an instrumented timing for this phase
  void setTiming(double time);

  /// Check to see if we are in the shutdown process, and handle it appropriately.
  void checkForShutdown();

  /// Start shutdown procedures for the controlPoints module(s). CkExit will be called once all outstanding operations have completed (e.g. waiting for idle time & memory usage to be gathered)
  void exitIfReady();

  /// All outstanding operations have completed, so do the shutdown now. First write files to disk, and then call CkExit().
  void doExitNow();

  /// Write data to disk (when needed), likely at exit
  void writeOutputToDisk();


  /// Entry method called on all PEs to request memory usage
  void requestMemoryUsage(CkCallback cb);
  /// All processors reduce their memory usages to this method
  void gatherMemoryUsage(CkReductionMsg *msg);


  /// Entry method called on all PEs to request memory usage
  void requestIdleTime(CkCallback cb);
  /// All processors reduce their memory usages in requestIdleTime() to this method
  void gatherIdleTime(CkReductionMsg *msg);
  

  /// Entry method called on all PEs to request Idle, Overhead, and Memory measurements
  void requestAll(CkCallback cb);
  /// All processors reduce their memory usages in requestIdleTime() to this method
  void gatherAll(CkReductionMsg *msg);
  

/*   /// Inform the control point framework that a named control point affects the priorities of some array   */
/*   void associatePriorityArray(const char *name, int groupIdx); */
  
/*   /// Inform the control point framework that a named control point affects the priority of some entry method */
/*   void associatePriorityEntry(const char *name, int idx); */
  


};

#endif

/** @} */
#endif

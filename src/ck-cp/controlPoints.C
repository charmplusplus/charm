#include <charm++.h>

// This file is compiled twice to make a version that is capable of not needing the tracing to be turned on. 

#include "controlPoints.h"
#include "trace-controlPoints.h"
#include "controlPoints.h"
#include "charm++.h"
#include "trace-projections.h"
#include <pathHistory.h>
#include "cp_effects.h"
#include <iostream>
#include <math.h>
#include <climits>

#if CMK_WITH_CONTROLPOINT

#define roundDouble(x)        ((long)(x+0.5))
#define myAbs(x)   (((x)>=0.0)?(x):(-1.0*(x)))
#define isInRange(v,a,b) ( ((v)<=(a)&&(v)>=(b)) || ((v)<=(b)&&(v)>=(a)) )

inline double closestInRange(double v, double a, double b){
  return (v<a) ? a : ((v>b)?b:v);
}


//  A framework for tuning "control points" exposed by an application. Tuning decisions are based upon observed performance measurements.
 

/** @defgroup ControlPointFramework Automatic Performance Tuning and Steering Framework  */
/**  @{ */

using namespace std;

#define DEFAULT_CONTROL_POINT_SAMPLE_PERIOD  10000000


//#undef DEBUGPRINT
//#define DEBUGPRINT 4

static void periodicProcessControlPoints(void* ptr, double currWallTime);


// A pointer to this PE's controlpoint manager Proxy
/* readonly */ CProxy_controlPointManager controlPointManagerProxy;
/* readonly */ int random_seed;
/* readonly */ long controlPointSamplePeriod;
/* readonly */ int whichTuningScheme;
/* readonly */ bool writeDataFileAtShutdown;
/* readonly */ bool shouldFilterOutputData;
/* readonly */ bool loadDataFileAtStartup;
/* readonly */ bool shouldGatherMemoryUsage;
/* readonly */ bool shouldGatherUtilization;
/* readonly */ bool shouldGatherAll;
/* readonly */ char CPDataFilename[512];

extern bool enableCPTracing;

/// The control point values to be used for the first few phases if the strategy doesn't choose to do something else.
/// These probably come from the command line arguments, so are available only on PE 0
std::map<std::string, int> defaultControlPointValues;



typedef enum tuningSchemeEnum {RandomSelection, SimulatedAnnealing, ExhaustiveSearch, CriticalPathAutoPrioritization, UseBestKnownTiming, UseSteering, MemoryAware, Simplex, DivideAndConquer, AlwaysDefaults, LDBPeriod, LDBPeriodLinear, LDBPeriodQuadratic, LDBPeriodOptimal}  tuningScheme;



void printTuningScheme(){
  switch(whichTuningScheme){
  case RandomSelection:
    CkPrintf("Tuning Scheme: RandomSelection\n");
    break;
  case AlwaysDefaults:
    CkPrintf("Tuning Scheme: AlwaysDefaults\n");
    break;
  case SimulatedAnnealing:
    CkPrintf("Tuning Scheme: SimulatedAnnealing\n");
    break;
  case ExhaustiveSearch:
    CkPrintf("Tuning Scheme: ExhaustiveSearch\n");
    break;
  case CriticalPathAutoPrioritization:
    CkPrintf("Tuning Scheme: CriticalPathAutoPrioritization\n");
    break;
  case UseBestKnownTiming:
    CkPrintf("Tuning Scheme: UseBestKnownTiming\n");
    break;
  case UseSteering:
    CkPrintf("Tuning Scheme: UseSteering\n");
    break;
  case MemoryAware:
    CkPrintf("Tuning Scheme: MemoryAware\n");
    break;
  case Simplex:
    CkPrintf("Tuning Scheme: Simplex Algorithm\n");
    break;
  case DivideAndConquer:
    CkPrintf("Tuning Scheme: Divide & Conquer Algorithm\n");
    break;
  case LDBPeriod:
    CkPrintf("Tuning Scheme: Load Balancing Period Steering (Constant Prediction)\n");
    break;
  case LDBPeriodLinear:
    CkPrintf("Tuning Scheme: Load Balancing Period Steering (Linear Prediction)\n");
    break;
  case LDBPeriodQuadratic:
    CkPrintf("Tuning Scheme: Load Balancing Period Steering (Quadratic Prediction)\n");
    break;
  case LDBPeriodOptimal:
    CkPrintf("Tuning Scheme: Load Balancing Period Steering (Optimal Prediction)\n");
    break;
  default:
    CkPrintf("Unknown tuning scheme\n");
    break;
  }
  fflush(stdout);
}



/// A reduction type that combines idle time measurements (min/sum/max etc.)
CkReduction::reducerType idleTimeReductionType;
/// A reducer that combines idle time measurements (min/sum/max etc.)
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



/// A reduction type that combines idle, overhead, and memory measurements
CkReduction::reducerType allMeasuresReductionType;
/// A reducer that combines idle, overhead, and memory measurements
#define ALL_REDUCTION_SIZE 12
CkReductionMsg *allMeasuresReduction(int nMsg,CkReductionMsg **msgs){
  double ret[ALL_REDUCTION_SIZE];
  if(nMsg > 0){
    CkAssert(msgs[0]->getSize()==ALL_REDUCTION_SIZE*sizeof(double));
    double *m=(double *)msgs[0]->getData();
    memcpy(ret, m, ALL_REDUCTION_SIZE*sizeof(double) );
  }
  for (int i=1;i<nMsg;i++) {
    CkAssert(msgs[i]->getSize()==ALL_REDUCTION_SIZE*sizeof(double));
    double *m=(double *)msgs[i]->getData();
    // idle time (min/sum/max)
    ret[0]=min(ret[0],m[0]);
    ret[1]+=m[1];
    ret[2]=max(ret[2],m[2]);
    // overhead time (min/sum/max)
    ret[3]=min(ret[3],m[3]);
    ret[4]+=m[4];
    ret[5]=max(ret[5],m[5]);
    // mem usage (max)
    ret[6]=max(ret[6],m[6]);
    // bytes per invocation for two types of entry methods
    ret[7]+=m[7];
    ret[8]+=m[8];
    ret[9]+=m[9];
    ret[10]+=m[10];
    // Grain size (avg)
    ret[11]+=m[11];
  }  
  return CkReductionMsg::buildNew(ALL_REDUCTION_SIZE*sizeof(double),ret);   
}


/// Registers the control point framework's reduction handlers at startup on each PE
/*initproc*/ void registerCPReductions(void) {
  idleTimeReductionType=CkReduction::addReducer(idleTimeReduction, false, "idleTimeReduction");
  allMeasuresReductionType=CkReduction::addReducer(allMeasuresReduction, false, "allMeasuresReduction");
}






/// Return an integer between 0 and num-1 inclusive
/// If different seed, name, and random_seed values are provided, the returned values are pseudo-random
unsigned int randInt(unsigned int num, const char* name, int seed=0){
  CkAssert(num > 0);

  unsigned long hash = 0;
  unsigned int c;
  unsigned char * str = (unsigned char*)name;

  while ((c = *str++)){
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



controlPointManager::controlPointManager() {
  generatedPlanForStep = -1;

    exitWhenReady = false;
    alreadyRequestedMemoryUsage = false;   
    alreadyRequestedIdleTime = false;
    alreadyRequestedAll = false;
    
    instrumentedPhase * newPhase = new instrumentedPhase();
    allData.phases.push_back(newPhase);   
    
    frameworkShouldAdvancePhase = false;
    haveControlPointChangeCallback = false;
//    CkPrintf("[%d] controlPointManager() Constructor Initializing control points, and loading data file\n", CkMyPe());
    
    phase_id = 0;

    if(loadDataFileAtStartup){    
      loadDataFile();
    }

    
    if(CkMyPe() == 0){
      CcdCallFnAfterOnPE((CcdVoidFn)periodicProcessControlPoints, (void*)NULL, controlPointSamplePeriod, CkMyPe());
    }

    traceRegisterUserEvent("No currently executing message", 5000);
    traceRegisterUserEvent("Zero time along critical path", 5010);
    traceRegisterUserEvent("Positive total time along critical path", 5020);
    traceRegisterUserEvent("env->setPathHistory()", 6000);
    traceRegisterUserEvent("Critical Path", 5900);
    traceRegisterUserEvent("Table Entry", 5901);


#if USER_EVENT_FOR_REGISTERTERMINALPATH
    traceRegisterUserEvent("registerTerminalPath", 100); 
#endif

  }
  

 controlPointManager::~controlPointManager(){
//    CkPrintf("[%d] controlPointManager() Destructor\n", CkMyPe());
  }

  void controlPointManager::pup(PUP::er &p)
  {
      // FIXME: does not work when control point is actually used,
      // just minimal pup so that it allows exit function to work (exitIfReady).
    p|generatedPlanForStep;
    p|exitWhenReady;
    p|alreadyRequestedMemoryUsage;
    p|alreadyRequestedIdleTime;
    p|alreadyRequestedAll;
    p|frameworkShouldAdvancePhase;
    p|haveControlPointChangeCallback;
    p|phase_id;
  }
  

  /// Loads the previous run data file
  void controlPointManager::loadDataFile(){
    ifstream infile(CPDataFilename);
    vector<std::string> names;
    std::string line;
  
    while(getline(infile,line)){
      if(line[0] != '#')
	break;
    }
  
    int numTimings = 0;
    std::istringstream n(line);
    n >> numTimings;
  
    while(getline(infile,line)){ 
      if(line[0] != '#') 
	break; 
    }

    int numControlPointNames = 0;
    std::istringstream n2(line);
    n2 >> numControlPointNames;
  
    for(int i=0; i<numControlPointNames; i++){
      getline(infile,line);
      names.push_back(line);
    }

    for(int i=0;i<numTimings;i++){
      getline(infile,line);
      while(line[0] == '#')
	getline(infile,line); 

      instrumentedPhase * ips = new instrumentedPhase();

      std::istringstream iss(line);

      // Read memory usage for phase
      iss >> ips->memoryUsageMB;
      //     CkPrintf("Memory usage loaded from file: %d\n", ips.memoryUsageMB);

      // Read idle time
      iss >> ips->idleTime.min;
      iss >> ips->idleTime.avg;
      iss >> ips->idleTime.max;

      // Read overhead time
      iss >> ips->overheadTime.min;
      iss >> ips->overheadTime.avg;
      iss >> ips->overheadTime.max;

      // read bytePerInvoke
      iss >> ips->bytesPerInvoke;

      // read grain size
      iss >> ips->grainSize;

      // Read control point values
      for(int cp=0;cp<numControlPointNames;cp++){
	int cpvalue;
	iss >> cpvalue;
	ips->controlPoints.insert(make_pair(names[cp],cpvalue));
      }

      // ignore median time
      double mt;
      iss >> mt;

      // Read all times

      double time;

      while(iss >> time){
	ips->times.push_back(time);
#if DEBUGPRINT > 5
	CkPrintf("read time %lf from file\n", time);
#endif
      }

      allData.phases.push_back(ips);

    }

    infile.close();
  }



  /// Add the current data to allData and output it to a file
  void controlPointManager::writeDataFile(){
    CkPrintf("============= writeDataFile() to %s  ============\n", CPDataFilename);
    ofstream outfile(CPDataFilename);
    allData.cleanupNames();

//    string s = allData.toString();
//    CkPrintf("At end: \n %s\n", s.c_str());

    if(shouldFilterOutputData){
      allData.verify();
      allData.filterOutIncompletePhases();
    }

//    string s2 = allData.toString();
//    CkPrintf("After filtering: \n %s\n", s2.c_str());
    if(allData.toString().length() > 10){
      outfile << allData.toString();
    } else {
      outfile << " No data available to save to disk " << endl;
    }
    outfile.close();
  }

  /// User can register a callback that is called when application should advance to next phase
  void controlPointManager::setCPCallback(CkCallback cb, bool _frameworkShouldAdvancePhase){
    frameworkShouldAdvancePhase = _frameworkShouldAdvancePhase;
    controlPointChangeCallback = cb;
    haveControlPointChangeCallback = true;
  }


/// A user can specify that the framework should advance the phases automatically. Useful for gather performance measurements without modifying a program.
void controlPointManager::setFrameworkAdvancePhase(bool _frameworkShouldAdvancePhase){
  frameworkShouldAdvancePhase = _frameworkShouldAdvancePhase;
}

  /// Called periodically by the runtime to handle the control points
  /// Currently called on each PE
  void controlPointManager::processControlPoints(){

#if DEBUGPRINT
    CkPrintf("[%d] processControlPoints() haveControlPointChangeCallback=%d frameworkShouldAdvancePhase=%d\n", CkMyPe(), (int)haveControlPointChangeCallback, (int)frameworkShouldAdvancePhase);
#endif

    //==========================================================================================
    // Print the data for each phase

    const int s = allData.phases.size();

#if DEBUGPRINT
    CkPrintf("\n\nExamining critical paths and priorities and idle times (num phases=%d)\n", s );
    for(int p=0;p<s;++p){
      const instrumentedPhase &phase = allData.phases[p];
      const idleTimeContainer &idle = phase.idleTime;
      //      vector<MergeablePathHistory> const &criticalPaths = phase.criticalPaths;
      vector<double> const &times = phase.times;

      CkPrintf("Phase %d:\n", p);
      idle.print();
     //   CkPrintf("critical paths: (* affected by control point)\n");
	//   for(int i=0;i<criticalPaths.size(); i++){
	// If affected by a control point
	//	criticalPaths[i].print();

      //	criticalPaths[i].print();
      //	CkPrintf("\n");
	

      //   }
      CkPrintf("Timings:\n");
      for(int i=0;i<times.size(); i++){
	CkPrintf("%lf ", times[i]);
      }
      CkPrintf("\n");

    }
    
    CkPrintf("\n\n");


#endif



    //==========================================================================================
    // If this is a phase during which we try to adapt control point values based on critical path
#if 0
    if( s%5 == 4) {

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
	CkAssert(phase.criticalPaths.size() > 0);
        CkAssert(phase.criticalPaths.size() > medianCriticalPathIdx); 
	
	CkPrintf("Median Critical Path has time %lf\n", path.getTotalTime());
	
	if(phase.times[medianCriticalPathIdx] > 1.2 * path.getTotalTime()){
	  CkPrintf("The application step(%lf) is taking significantly longer than the critical path(%lf). BAD\n",phase.times[medianCriticalPathIdx], path.getTotalTime() );


	  CkPrintf("Finding control points related to the critical path\n");
	  int cpcount = 0;
	  std::set<std::string> controlPointsAffectingCriticalPath;

	  
	  for(int e=0;e<path.getNumUsed();e++){
	    if(path.getUsedCount(e)>0){
	      int ep = path.getUsedEp(e);

	      std::map<std::string, std::set<int> >::iterator iter;
	      for(iter=affectsPrioritiesEP.begin(); iter!= affectsPrioritiesEP.end(); ++iter){
		if(iter->second.count(ep)>0){
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
	    
	    if(haveControlPointChangeCallback){ 
#if DEBUGPRINT
	      CkPrintf("Calling control point change callback\n");
#endif
	      // Create a high priority message and send it to the callback
	      controlPointMsg *msg = new (8*sizeof(int)) controlPointMsg; 
	      *((int*)CkPriorityPtr(msg)) = -INT_MAX;
	      CkSetQueueing(msg, CK_QUEUEING_IFIFO);
	      controlPointChangeCallback.send(msg);
	      
	    }
	    
	    
	    // adjust the control points that can affect the critical path

	    char textDescription[4096*2];
	    textDescription[0] = '\0';

	    std::map<std::string,int>::iterator newCP;
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
	      std::map<std::string,int>::iterator newCP;
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
      
    } else {
      
      
    }
   
#endif



    if(frameworkShouldAdvancePhase){
      gotoNextPhase();	
    }
    
    if(haveControlPointChangeCallback){ 
      // Create a high priority message and send it to the callback
      controlPointMsg *msg = new (8*sizeof(int)) controlPointMsg; 
      *((int*)CkPriorityPtr(msg)) = -INT_MAX;
      CkSetQueueing(msg, CK_QUEUEING_IFIFO);
      controlPointChangeCallback.send(msg);
    }
    
  }
  
  /// Determine if any control point is known to affect an entry method
  bool controlPointManager::controlPointAffectsThisEP(int ep){
    std::map<std::string, std::set<int> >::iterator iter;
    for(iter=affectsPrioritiesEP.begin(); iter!= affectsPrioritiesEP.end(); ++iter){
      if(iter->second.count(ep)>0){
	return true;
      }
    }
    return false;    
  }
  
  /// Determine if any control point is known to affect a chare array  
  bool controlPointManager::controlPointAffectsThisArray(int array){
    std::map<std::string, std::set<int> >::iterator iter;
    for(iter=affectsPrioritiesArray.begin(); iter!= affectsPrioritiesArray.end(); ++iter){
      if(iter->second.count(array)>0){
	return true;
      }
    }
    return false;   
  }
  

  /// The data from the current phase
  instrumentedPhase * controlPointManager::currentPhaseData(){
    int s = allData.phases.size();
    CkAssert(s>=1);
    return allData.phases[s-1];
  }
 

  /// The data from the previous phase
  instrumentedPhase * controlPointManager::previousPhaseData(){
    int s = allData.phases.size();
    if(s >= 2 && phase_id > 0) {
      return allData.phases[s-2];
    } else {
      return NULL;
    }
  }
 
  /// The data from two phases back
  instrumentedPhase * controlPointManager::twoAgoPhaseData(){
    int s = allData.phases.size();
    if(s >= 3 && phase_id > 0) {
      return allData.phases[s-3];
    } else {
      return NULL;
    }
  }
  

  /// Called by either the application or the Control Point Framework to advance to the next phase  
  void controlPointManager::gotoNextPhase(){
    CkPrintf("gotoNextPhase shouldGatherAll=%d enableCPTracing=%d\n", (int)shouldGatherAll, (int)enableCPTracing);
    fflush(stdout);
      
    if(enableCPTracing){
      if(shouldGatherAll && CkMyPe() == 0 && !alreadyRequestedAll){
	alreadyRequestedAll = true;
	CkCallback *cb = new CkCallback(CkIndex_controlPointManager::gatherAll(NULL), 0, thisProxy);
	CkPrintf("Requesting all measurements\n");
	thisProxy.requestAll(*cb);
	delete cb;
	
      } else {
	
	if(shouldGatherMemoryUsage && CkMyPe() == 0 && !alreadyRequestedMemoryUsage){
	  alreadyRequestedMemoryUsage = true;
	  CkCallback *cb = new CkCallback(CkIndex_controlPointManager::gatherMemoryUsage(NULL), 0, thisProxy);
	  thisProxy.requestMemoryUsage(*cb);
	  delete cb;
	}
	
	if(shouldGatherUtilization && CkMyPe() == 0 && !alreadyRequestedIdleTime){
	  alreadyRequestedIdleTime = true;
	  CkCallback *cb = new CkCallback(CkIndex_controlPointManager::gatherIdleTime(NULL), 0, thisProxy);
	  thisProxy.requestIdleTime(*cb);
	  delete cb;
	}
      }
    }

    // increment phase id
    phase_id++;
    

    // Create new entry for the phase we are starting now
    instrumentedPhase * newPhase = new instrumentedPhase();
    allData.phases.push_back(newPhase);
    
    CkPrintf("Now in phase %d allData.phases.size()=%zu\n", phase_id, allData.phases.size());

  }

  /// An application uses this to register an instrumented timing for this phase
  void controlPointManager::setTiming(double time){
    currentPhaseData()->times.push_back(time);

#if USE_CRITICAL_PATH_HEADER_ARRAY
       
    // First we should register this currently executing message as a path, because it is likely an important one to consider.
    //    registerTerminalEntryMethod();
    
    // save the critical path for this phase
    //   thisPhaseData.criticalPaths.push_back(maxTerminalPathHistory);
    //    maxTerminalPathHistory.reset();
    
    
    // Reset the counts for the currently executing message
    //resetThisEntryPath();
        
#endif
    
  }
  
  /// Entry method called on all PEs to request CPU utilization statistics
  void controlPointManager::requestIdleTime(CkCallback cb){
    CkAssert(enableCPTracing);
   
    double i = localControlPointTracingInstance()->idleRatio();
    double idle[3];
    idle[0] = i;
    idle[1] = i;
    idle[2] = i;
    
    //    CkPrintf("[%d] idleRatio=%f\n", CkMyPe(), i);
    
    localControlPointTracingInstance()->resetTimings();
    
    contribute(3*sizeof(double),idle,idleTimeReductionType, cb);
  }
  
  /// All processors reduce their memory usages in requestIdleTime() to this method
  void controlPointManager::gatherIdleTime(CkReductionMsg *msg){
    CkAssert(enableCPTracing);

    int size=msg->getSize() / sizeof(double);
    CkAssert(size==3);
    double *r=(double *) msg->getData();
        
    instrumentedPhase* prevPhase = previousPhaseData();
    if(prevPhase != NULL){
      prevPhase->idleTime.min = r[0];
      prevPhase->idleTime.avg = r[1]/CkNumPes();
      prevPhase->idleTime.max = r[2];
      prevPhase->idleTime.print();
      CkPrintf("Stored idle time min=%lf avg=%lf max=%lf in prevPhase=%p\n", prevPhase->idleTime.min, prevPhase->idleTime.avg, prevPhase->idleTime.max, prevPhase);
    } else {
      CkPrintf("There is no previous phase to store the idle time measurements\n");
    }
    
    alreadyRequestedIdleTime = false;
    checkForShutdown();
    delete msg;
  }






  /// Entry method called on all PEs to request CPU utilization statistics and memory usage
  void controlPointManager::requestAll(CkCallback cb){
    CkAssert(enableCPTracing);

    TraceControlPoints *t = localControlPointTracingInstance();

    double data[ALL_REDUCTION_SIZE];

    double *idle = data;
    double *over = data+3;
    double *mem = data+6;
    double *msgBytes = data+7;  
    double *grainsize = data+11;  

    const double i = t->idleRatio();
    idle[0] = i;
    idle[1] = i;
    idle[2] = i;

    const double o = t->overheadRatio();
    over[0] = o;
    over[1] = o;
    over[2] = o;

    const double m = t->memoryUsageMB();
    mem[0] = m;

    msgBytes[0] = t->b2;
    msgBytes[1] = t->b3;
    msgBytes[2] = t->b2mlen;
    msgBytes[3] = t->b3mlen;

    grainsize[0] = t->grainSize();
    
    localControlPointTracingInstance()->resetAll();

    contribute(ALL_REDUCTION_SIZE*sizeof(double),data,allMeasuresReductionType, cb);
  }
  
  /// All processors reduce their memory usages in requestIdleTime() to this method
  void controlPointManager::gatherAll(CkReductionMsg *msg){
    CkAssert(enableCPTracing);

    CkAssert(msg->getSize()==ALL_REDUCTION_SIZE*sizeof(double));
    int size=msg->getSize() / sizeof(double);
    double *data=(double *) msg->getData();
        
    double *idle = data;
    double *over = data+3;
    double *mem = data+6;
    double *msgBytes = data+7;
    double *granularity = data+11;


    //    std::string b = allData.toString();

    instrumentedPhase* prevPhase = previousPhaseData();
    if(prevPhase != NULL){
      prevPhase->idleTime.min = idle[0];
      prevPhase->idleTime.avg = idle[1]/CkNumPes();
      prevPhase->idleTime.max = idle[2];
      
      prevPhase->overheadTime.min = over[0];
      prevPhase->overheadTime.avg = over[1]/CkNumPes();
      prevPhase->overheadTime.max = over[2];
      
      prevPhase->memoryUsageMB = mem[0];
      
      CkPrintf("Stored idle time min=%lf avg=%lf max=%lf  mem=%lf in prevPhase=%p\n", (double)prevPhase->idleTime.min, prevPhase->idleTime.avg, prevPhase->idleTime.max, (double)prevPhase->memoryUsageMB, prevPhase);

      double bytesPerInvoke2 = msgBytes[2] / msgBytes[0];
      double bytesPerInvoke3 = msgBytes[3] / msgBytes[1];

      /** The average of the grain sizes on all PEs in us */
      prevPhase->grainSize = (granularity[0] / (double)CkNumPes());

      CkPrintf("Bytes Per Invokation 2 = %f\n", bytesPerInvoke2);
      CkPrintf("Bytes Per Invokation 3 = %f\n", bytesPerInvoke3);

      CkPrintf("Bytes Per us of work 2 = %f\n", bytesPerInvoke2/prevPhase->grainSize);
      CkPrintf("Bytes Per us of work 3 = %f\n", bytesPerInvoke3/prevPhase->grainSize);

      if(bytesPerInvoke2 > bytesPerInvoke3)
	prevPhase->bytesPerInvoke = bytesPerInvoke2;
      else
	prevPhase->bytesPerInvoke = bytesPerInvoke3;

      //      prevPhase->print();
      // CkPrintf("prevPhase=%p number of timings=%d\n", prevPhase, prevPhase->times.size() );

      //      std::string a = allData.toString();

      //CkPrintf("Before:\n%s\nAfter:\n%s\n\n", b.c_str(), a.c_str());
      
    } else {
      CkPrintf("There is no previous phase to store measurements\n");
    }
    
    alreadyRequestedAll = false;
    checkForShutdown();
    delete msg;
  }




  void controlPointManager::checkForShutdown(){
    if( exitWhenReady && !alreadyRequestedAll && !alreadyRequestedMemoryUsage && !alreadyRequestedIdleTime && CkMyPe()==0){
      doExitNow();
    }
  }


  void controlPointManager::exitIfReady(){
     if( !alreadyRequestedMemoryUsage && !alreadyRequestedAll && !alreadyRequestedIdleTime && CkMyPe()==0){
       //  CkPrintf("controlPointManager::exitIfReady exiting immediately\n");
       doExitNow();
     } else {
       // CkPrintf("controlPointManager::exitIfReady Delaying exiting\n");
       exitWhenReady = true;
     }
  }



  void controlPointManager::doExitNow(){
          _TRACE_BEGIN_EXECUTE_DETAILED(-1, -1, _threadEP,CkMyPe(), 0, NULL, this);
	  writeOutputToDisk();
    // CkPrintf("[%d] Control point manager calling CkContinueExit()\n", CkMyPe());
    CkContinueExit();
  }

  void controlPointManager::writeOutputToDisk(){
	  if(writeDataFileAtShutdown){
		  controlPointManagerProxy.ckLocalBranch()->writeDataFile();
	  }
  }


  /// Entry method called on all PEs to request memory usage
  void controlPointManager::requestMemoryUsage(CkCallback cb){
    int m = CmiMaxMemoryUsage() / 1024 / 1024;
    CmiResetMaxMemory();
    //    CkPrintf("PE %d Memory Usage is %d MB\n",CkMyPe(), m);
    contribute(sizeof(int),&m,CkReduction::max_int, cb);
  }

  /// All processors reduce their memory usages to this method
  void controlPointManager::gatherMemoryUsage(CkReductionMsg *msg){
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
    checkForShutdown();
    delete msg;
  }


//   /// Inform the control point framework that a named control point affects the priorities of some array  
//   void controlPointManager::associatePriorityArray(const char *name, int groupIdx){
//     CkPrintf("Associating control point \"%s\" affects priority of array id=%d\n", name, groupIdx );
    
//     if(affectsPrioritiesArray.count(std::string(name)) > 0 ) {
//       affectsPrioritiesArray[std::string(name)].insert(groupIdx);
//     } else {
//       std::set<int> s;
//       s.insert(groupIdx);
//       affectsPrioritiesArray[std::string(name)] = s;
//     }
    
// #if DEBUGPRINT   
//     std::map<std::string, std::set<int> >::iterator f;
//     for(f=affectsPrioritiesArray.begin(); f!=affectsPrioritiesArray.end();++f){
//       std::string name = f->first;
//       std::set<int> &vals = f->second;
//       cout << "Control point " << name << " affects arrays: ";
//       std::set<int>::iterator i;
//       for(i=vals.begin(); i!=vals.end();++i){
// 	cout << *i << " ";
//       }
//       cout << endl;
//     }
// #endif
    
//   }
  
//   /// Inform the control point framework that a named control point affects the priority of some entry method
//   void controlPointManager::associatePriorityEntry(const char *name, int idx){
//     CkPrintf("Associating control point \"%s\" with EP id=%d\n", name, idx);

//       if(affectsPrioritiesEP.count(std::string(name)) > 0 ) {
//       affectsPrioritiesEP[std::string(name)].insert(idx);
//     } else {
//       std::set<int> s;
//       s.insert(idx);
//       affectsPrioritiesEP[std::string(name)] = s;
//     }
    
// #if DEBUGPRINT
//     std::map<std::string, std::set<int> >::iterator f;
//     for(f=affectsPrioritiesEP.begin(); f!=affectsPrioritiesEP.end();++f){
//       std::string name = f->first;
//       std::set<int> &vals = f->second;
//       cout << "Control point " << name << " affects EP: ";
//       std::set<int>::iterator i;
//       for(i=vals.begin(); i!=vals.end();++i){
// 	cout << *i << " ";
//       }
//       cout << endl;
//     }
// #endif


//   }
  


/// An interface callable by the application.
void gotoNextPhase(){
  controlPointManagerProxy.ckLocalBranch()->gotoNextPhase();
}

FLINKAGE void FTN_NAME(GOTONEXTPHASE,gotonextphase)()
{
  gotoNextPhase();
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
    
    
    double period, periodms;
    bool haveSamplePeriod = CmiGetArgDoubleDesc(args->argv,"+CPSamplePeriod", &period,"The time between Control Point Framework samples (in seconds)");
    bool haveSamplePeriodMs = CmiGetArgDoubleDesc(args->argv,"+CPSamplePeriodMs", &periodms,"The time between Control Point Framework samples (in milliseconds)");
    if(haveSamplePeriod){
      CkPrintf("controlPointSamplePeriod = %lf sec\n", period);
      controlPointSamplePeriod =  (int)(period * 1000); /**< A readonly */
    } else if(haveSamplePeriodMs){
      CkPrintf("controlPointSamplePeriodMs = %lf ms\n", periodms);
      controlPointSamplePeriod = periodms; /**< A readonly */
    } else {
      controlPointSamplePeriod =  DEFAULT_CONTROL_POINT_SAMPLE_PERIOD;
    }

  
    
    whichTuningScheme = RandomSelection;


    if( CmiGetArgFlagDesc(args->argv,"+CPSchemeRandom","Randomly Select Control Point Values") ){
      whichTuningScheme = RandomSelection;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPExhaustiveSearch","Exhaustive Search of Control Point Values") ){
      whichTuningScheme = ExhaustiveSearch;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPAlwaysUseDefaults","Always Use The Provided Default Control Point Values") ){
      whichTuningScheme = AlwaysDefaults;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPSimulAnneal","Simulated Annealing Search of Control Point Values") ){
      whichTuningScheme = SimulatedAnnealing;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPCriticalPathPrio","Use Critical Path to adapt Control Point Values") ){
      whichTuningScheme = CriticalPathAutoPrioritization;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPBestKnown","Use BestKnown Timing for Control Point Values") ){
      whichTuningScheme = UseBestKnownTiming;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPSteering","Use Steering to adjust Control Point Values") ){
      whichTuningScheme = UseSteering;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPMemoryAware", "Adjust control points to approach available memory") ){
      whichTuningScheme = MemoryAware;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPSimplex", "Nelder-Mead Simplex Algorithm") ){
      whichTuningScheme = Simplex;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPDivideConquer", "A divide and conquer program specific steering scheme") ){
      whichTuningScheme = DivideAndConquer;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPLDBPeriod", "Adjust the load balancing period (Constant Predictor)") ){
      whichTuningScheme = LDBPeriod;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPLDBPeriodLinear", "Adjust the load balancing period (Linear Predictor)") ){
      whichTuningScheme = LDBPeriodLinear;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPLDBPeriodQuadratic", "Adjust the load balancing period (Quadratic Predictor)") ){
      whichTuningScheme = LDBPeriodQuadratic;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPLDBPeriodOptimal", "Adjust the load balancing period (Optimal Predictor)") ){
      whichTuningScheme = LDBPeriodOptimal;
    }

    char *defValStr = NULL;
    if( CmiGetArgStringDesc(args->argv, "+CPDefaultValues", &defValStr, "Specify the default control point values used for the first couple phases") ){
      CkPrintf("You specified default value string: %s\n", defValStr);
      
      // Parse the string, looking for commas
     

      char *tok = strtok(defValStr, ",");
      while (tok) {
	// split on the equal sign
	int len = strlen(tok);
	char *eqsign = strchr(tok, '=');
	if(eqsign != NULL && eqsign != tok){
	  *eqsign = '\0';
	  char *cpname = tok;
	  std::string cpName(tok);
	  char *cpDefVal = eqsign+1;	  
	  int v=-1;
	  if(sscanf(cpDefVal, "%d", &v) == 1){
	    CkPrintf("Command Line Argument Specifies that Control Point \"%s\" defaults to %d\n", cpname, v);
	    CkAssert(CkMyPe() == 0); // can only access defaultControlPointValues on PE 0
	    defaultControlPointValues[cpName] = v;
	  }
	}
	tok = strtok(NULL, ",");
      }

    }

    shouldGatherAll = false;
    shouldGatherMemoryUsage = false;
    shouldGatherUtilization = false;
    
    if ( CmiGetArgFlagDesc(args->argv,"+CPGatherAll","Gather all types of measurements for each phase") ){
      shouldGatherAll = true;
    } else {
      if ( CmiGetArgFlagDesc(args->argv,"+CPGatherMemoryUsage","Gather memory usage after each phase") ){
	shouldGatherMemoryUsage = true;
      }
      if ( CmiGetArgFlagDesc(args->argv,"+CPGatherUtilization","Gather utilization & Idle time after each phase") ){
	shouldGatherUtilization = true;
      }
    }
    
    writeDataFileAtShutdown = false;   
    if( CmiGetArgFlagDesc(args->argv,"+CPSaveData","Save Control Point timings & configurations at completion") ){
      writeDataFileAtShutdown = true;
    }

    shouldFilterOutputData = true;
    if( CmiGetArgFlagDesc(args->argv,"+CPNoFilterData","Don't filter phases from output data") ){
      shouldFilterOutputData = false;
    }


   loadDataFileAtStartup = false;   
    if( CmiGetArgFlagDesc(args->argv,"+CPLoadData","Load Control Point timings & configurations at startup") ){
      loadDataFileAtStartup = true;
    }

    char *cpdatafile;
    if( CmiGetArgStringDesc(args->argv, "+CPDataFilename", &cpdatafile, "Specify control point data file to save/load") ){
      snprintf(CPDataFilename, sizeof(CPDataFilename), "%s", cpdatafile);
    } else {
      strcpy(CPDataFilename, "controlPointData.txt");
    }


    controlPointManagerProxy = CProxy_controlPointManager::ckNew();

    delete args;
  }
  ~controlPointMain(){}
};

/// An interface callable by the application.
void registerCPChangeCallback(CkCallback cb, bool frameworkShouldAdvancePhase){
  CkAssert(CkMyPe() == 0);
  CkPrintf("Application has registered a control point change callback\n");
  controlPointManagerProxy.ckLocalBranch()->setCPCallback(cb, frameworkShouldAdvancePhase);
}

/// An interface callable by the application.
void setFrameworkAdvancePhase(bool frameworkShouldAdvancePhase){
  if(CkMyPe() == 0){
    CkPrintf("Application has specified that framework should %sadvance phase\n", frameworkShouldAdvancePhase?"":"not ");
    controlPointManagerProxy.ckLocalBranch()->setFrameworkAdvancePhase(frameworkShouldAdvancePhase);
  }
}

/// An interface callable by the application.
void registerControlPointTiming(double time){
  CkAssert(CkMyPe() == 0);
#if DEBUGPRINT>0
  CkPrintf("Program registering its own timing with registerControlPointTiming(time=%lf)\n", time);
#endif
  controlPointManagerProxy.ckLocalBranch()->setTiming(time);
}

/// An interface callable by the application.
void controlPointTimingStamp() {
  CkAssert(CkMyPe() == 0);
#if DEBUGPRINT>0
  CkPrintf("Program registering its own timing with controlPointTimingStamp()\n", time);
#endif
  
  static double prev_time = 0.0;
  double t = CmiWallTimer();
  double duration = t - prev_time;
  prev_time = t;
    
  controlPointManagerProxy.ckLocalBranch()->setTiming(duration);
}

FLINKAGE void FTN_NAME(CONTROLPOINTTIMINGSTAMP,controlpointtimingstamp)()
{
  controlPointTimingStamp();
}


FLINKAGE void FTN_NAME(SETFRAMEWORKADVANCEPHASEF,setframeworkadvancephasef)(CMK_TYPEDEF_INT4 *value)
{
  setFrameworkAdvancePhase(*value);
}




/// Shutdown the control point framework, writing data to disk if necessary
extern "C" void controlPointShutdown(){
  if(CkMyPe() == 0){

    if (!controlPointManagerProxy.ckGetGroupID().isZero()) {
      // wait for gathering of idle time & memory usage to complete
      controlPointManagerProxy.ckLocalBranch()->exitIfReady();
    } else {
      CkContinueExit();
    }
  }
}

/// A function called at startup on each node to register controlPointShutdown() to be called at CkExit()
void controlPointInitNode(){
  registerExitFn(controlPointShutdown);
}

/// Called periodically to allow control point framework to do things periodically
static void periodicProcessControlPoints(void* ptr, double currWallTime){
#ifdef DEBUGPRINT
  CkPrintf("[%d] periodicProcessControlPoints()\n", CkMyPe());
#endif
  controlPointManagerProxy.ckLocalBranch()->processControlPoints();
  CcdCallFnAfterOnPE((CcdVoidFn)periodicProcessControlPoints, (void*)NULL, controlPointSamplePeriod, CkMyPe());
}





/// Determine a control point value using some optimization scheme (use max known, simmulated annealling, 
/// user observed characteristic to adapt specific control point values.
/// This function must return valid values for newControlPoints.
void controlPointManager::generatePlan() {
  const double startGenerateTime = CmiWallTimer();
  const int phase_id = this->phase_id;
  const int effective_phase = allData.phases.size();

  // Only generate a plan once per phase
  if(generatedPlanForStep == phase_id) 
    return;
  generatedPlanForStep = phase_id;
 
  CkPrintf("Generating Plan for phase %d\n", phase_id); 
  printTuningScheme();

  // By default lets put the previous phase data into newControlPoints
  instrumentedPhase *prevPhase = previousPhaseData();
  for(std::map<std::string, int >::const_iterator cpsIter=prevPhase->controlPoints.begin(); cpsIter != prevPhase->controlPoints.end(); ++cpsIter){
	  newControlPoints[cpsIter->first] = cpsIter->second;
  }


  if( whichTuningScheme == RandomSelection ){
    std::map<std::string, std::pair<int,int> >::const_iterator cpsIter;
    for(cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
      const std::string &name = cpsIter->first;
      const std::pair<int,int> &bounds = cpsIter->second;
      const int lb = bounds.first;
      const int ub = bounds.second;
      newControlPoints[name] = lb + randInt(ub-lb+1, name.c_str(), phase_id);
    }
  } else if( whichTuningScheme == CriticalPathAutoPrioritization) {
    // -----------------------------------------------------------
    //  USE CRITICAL PATH TO ADJUST PRIORITIES
    //   
    // This scheme will return the median value for the range 
    // early on, and then will transition over to the new control points
    // determined by the critical path adapting code

    // This won't work until the periodic function is fixed up a bit

    std::map<std::string, std::pair<int,int> >::const_iterator cpsIter;
    for(cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
      const std::string &name = cpsIter->first;
      const std::pair<int,int> &bounds = cpsIter->second;
      const int lb = bounds.first;
      const int ub = bounds.second;
      newControlPoints[name] =  (lb+ub)/2;
    }

  } else if ( whichTuningScheme == MemoryAware ) {

    // -----------------------------------------------------------
    //  STEERING BASED ON MEMORY USAGE

    instrumentedPhase *twoAgoPhase = twoAgoPhaseData();
    instrumentedPhase *prevPhase = previousPhaseData();
 
    if(phase_id%2 == 0){
      CkPrintf("Steering (memory based) based on 2 phases ago:\n");
      twoAgoPhase->print();
      CkPrintf("\n");
      fflush(stdout);
      
      // See if memory usage is low:
      double memUsage = twoAgoPhase->memoryUsageMB;
      CkPrintf("Steering (memory based) encountered memory usage of (%f MB)\n", memUsage);
      fflush(stdout);
      if(memUsage < 1100.0 && memUsage > 0.0){ // Kraken has about 16GB and 12 cores per node
	CkPrintf("Steering (memory based) encountered low memory usage (%f) < 1200 \n", memUsage);
	CkPrintf("Steering (memory based) controlPointSpace.size()=%zu\n", controlPointSpace.size());
	
	// Initialize plan to be the values from two phases ago (later we'll adjust this)
	newControlPoints = twoAgoPhase->controlPoints;


	CkPrintf("Steering (memory based) initialized plan\n");
	fflush(stdout);

	// look for a possible control point knob to turn
	std::map<std::string, std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > > &possibleCPsToTune = CkpvAccess(cp_effects)["MemoryConsumption"];
	
	// FIXME: assume for now that we just have one control point with the effect, and one direction to turn it
	bool found = false;
	std::string cpName;
	std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > *info;
	std::map<std::string, std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > >::iterator iter;
	for(iter = possibleCPsToTune.begin(); iter != possibleCPsToTune.end(); iter++){
	  cpName = iter->first;
	  info = &iter->second;
	  found = true;
	  break;
	}

	// Adapt the control point value
	if(found){
	  CkPrintf("Steering found knob to turn that should increase memory consumption\n");
	  fflush(stdout);
	  const int twoAgoValue =  twoAgoPhase->controlPoints[cpName];
	  const int maxValue = controlPointSpace[cpName].second;
	  
	  if(twoAgoValue+1 <= maxValue){
	    newControlPoints[cpName] = twoAgoValue+1; // increase from two phases back
	  }
	}
	
      }
    }

    CkPrintf("Steering (memory based) done for this phase\n");
    fflush(stdout);

  } else if ( whichTuningScheme == UseBestKnownTiming ) {

    // -----------------------------------------------------------
    //  USE BEST KNOWN TIME  
    static bool firstTime = true;
    if(firstTime){
      firstTime = false;
      instrumentedPhase *best = allData.findBest(); 
      CkPrintf("Best known phase is:\n");
      best->print();
      CkPrintf("\n");
      
      std::map<std::string, std::pair<int,int> >::const_iterator cpsIter;
      for(cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter) {
	const std::string &name = cpsIter->first;
	newControlPoints[name] =  best->controlPoints[name];
      }
    }
  } else if( whichTuningScheme == LDBPeriod) {
    // Assume this is used in this manner:
    //  1) go to next phase
    //  2) request control point
    //  3) load balancing
    //  4) computation
    
    
    instrumentedPhase *twoAgoPhase = twoAgoPhaseData();
    instrumentedPhase *prevPhase = previousPhaseData();
    
    
    const std::vector<double> &times = twoAgoPhase->times;
    const int oldNumTimings = times.size();


    const std::vector<double> &timesNew = prevPhase->times;
    const int newNumTimings = timesNew.size();


    if(oldNumTimings > 4 && newNumTimings > 4){
      
      // Build model of execution time based on two phases ago
      // Compute the average times for each 1/3 of the steps, except for the 2 first steps where load balancing occurs
      
      double oldSum = 0;
      
      for(int i=2; i<oldNumTimings; i++){
	oldSum += times[i];
      }
      
      const double oldAvg = oldSum / (oldNumTimings-2);
      
      
      
      
      // Computed as an integral from 0.5 to the number of bins of the same size as two ago phase + 0.5
      const double expectedTotalTime = oldAvg * newNumTimings;
      
      
      // Measure actual time
      double newSum = 0.0;
      for(int i=2; i<newNumTimings; ++i){
	newSum += timesNew[i];
      }
      
      const double newAvg = newSum / (newNumTimings-2);
      const double newTotalTimeExcludingLBSteps = newAvg * ((double)newNumTimings); // excluding the load balancing abnormal steps
      
      const double benefit = expectedTotalTime - newTotalTimeExcludingLBSteps;
      
      // Determine load balance cost
      const double lbcost = timesNew[0] + timesNew[1] - 2.0*newAvg;
      
      const double benefitAfterLB = benefit - lbcost;
    
    
      // Determine whether LB cost outweights the estimated benefit
      CkPrintf("Constant Model: lbcost = %f, expected = %f, actual = %f\n", lbcost, expectedTotalTime, newTotalTimeExcludingLBSteps);
    
    
      std::map<std::string, std::pair<int,int> >::const_iterator cpsIter;
      for(cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
	const std::string &name = cpsIter->first;
	const std::pair<int,int> &bounds = cpsIter->second;
	const int lb = bounds.first;
	const int ub = bounds.second;
      
	if(benefitAfterLB > 0){
	  CkPrintf("Constant Model: Beneficial LB\n");
	  int newval = newControlPoints[name] / 2;
	  if(newval > lb)
	    newControlPoints[name] = newval;
	  else 
	    newControlPoints[name] = lb;
	} else {
	  CkPrintf("Constant Model: Detrimental LB\n");
	  int newval = newControlPoints[name] * 2;
	  if(newval < ub)
	    newControlPoints[name] = newval;
	  else
	    newControlPoints[name] = ub;
	}
      }
    }
    
    
  }  else if( whichTuningScheme == LDBPeriodLinear) {
    // Assume this is used in this manner:
    //  1) go to next phase
    //  2) request control point
    //  3) load balancing
    //  4) computation


    instrumentedPhase *twoAgoPhase = twoAgoPhaseData();
    instrumentedPhase *prevPhase = previousPhaseData();
    
    const std::vector<double> &times = twoAgoPhase->times;
    const int oldNumTimings = times.size();

    const std::vector<double> &timesNew = prevPhase->times;
    const int newNumTimings = timesNew.size();
    

    if(oldNumTimings > 4 && newNumTimings > 4){

      // Build model of execution time based on two phases ago
      // Compute the average times for each 1/3 of the steps, except for the 2 first steps where load balancing occurs
      const int b1 = 2 + (oldNumTimings-2)/2;
      double s1 = 0;
      double s2 = 0;
    
      const double ldbStepsTime = times[0] + times[1];
    
      for(int i=2; i<b1; i++){
	s1 += times[i];
      }
      for(int i=b1; i<oldNumTimings; i++){
	s2 += times[i];
      }
      
      
      // Compute the estimated time for the last phase's data
    
      const double a1 = s1 / (double)(b1-2);
      const double a2 = s2 / (double)(oldNumTimings-b1);
      const double aold = (a1+a2)/2.0;

      const double expectedTotalTime = newNumTimings*(aold+(oldNumTimings+newNumTimings)*(a2-a1)/oldNumTimings);
        
    
      // Measure actual time
      double sum = 0.0;
      for(int i=2; i<newNumTimings; ++i){
	sum += timesNew[i];
      }

      const double avg = sum / ((double)(newNumTimings-2));
      const double actualTotalTime = avg * ((double)newNumTimings); // excluding the load balancing abnormal steps

      const double benefit = expectedTotalTime - actualTotalTime;

      // Determine load balance cost
      const double lbcost = timesNew[0] + timesNew[1] - 2.0*avg;

      const double benefitAfterLB = benefit - lbcost;

    
      // Determine whether LB cost outweights the estimated benefit
      CkPrintf("Linear Model: lbcost = %f, expected = %f, actual = %f\n", lbcost, expectedTotalTime, actualTotalTime);
    
    
    
      std::map<std::string, std::pair<int,int> >::const_iterator cpsIter;
      for(cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
	const std::string &name = cpsIter->first;
	const std::pair<int,int> &bounds = cpsIter->second;
	const int lb = bounds.first;
	const int ub = bounds.second;
      
	if(benefitAfterLB > 0){
	  CkPrintf("Linear Model: Beneficial LB\n");
	  int newval = newControlPoints[name] / 2;
	  if(newval > lb)
	    newControlPoints[name] = newval;
	  else 
	    newControlPoints[name] = lb;
	} else {
	  CkPrintf("Linear Model: Detrimental LB\n");
	  int newval = newControlPoints[name] * 2;
	  if(newval < ub)
	    newControlPoints[name] = newval;
	  else 
	    newControlPoints[name] = ub;
	}
      }
    }

  }

  else if( whichTuningScheme == LDBPeriodQuadratic) {
    // Assume this is used in this manner:
    //  1) go to next phase
    //  2) request control point
    //  3) load balancing
    //  4) computation


    instrumentedPhase *twoAgoPhase = twoAgoPhaseData();
    instrumentedPhase *prevPhase = previousPhaseData();
        
    const std::vector<double> &times = twoAgoPhase->times;
    const int oldNumTimings = times.size();

    const std::vector<double> &timesNew = prevPhase->times;
    const int newNumTimings = timesNew.size();

    
    if(oldNumTimings > 4 && newNumTimings > 4){


      // Build model of execution time based on two phases ago
      // Compute the average times for each 1/3 of the steps, except for the 2 first steps where load balancing occurs
      const int b1 = 2 + (oldNumTimings-2)/3;
      const int b2 = 2 + (2*(oldNumTimings-2))/3;
      double s1 = 0;
      double s2 = 0;
      double s3 = 0;

      const double ldbStepsTime = times[0] + times[1];
    
      for(int i=2; i<b1; i++){
	s1 += times[i];
      }
      for(int i=b1; i<b2; i++){
	s2 += times[i];
      }
      for(int i=b2; i<oldNumTimings; i++){
	s3 += times[i];
      }

    
      // Compute the estimated time for the last phase's data
    
      const double a1 = s1 / (double)(b1-2);
      const double a2 = s2 / (double)(b2-b1);
      const double a3 = s3 / (double)(oldNumTimings-b2);
    
      const double a = (a1-2.0*a2+a3)/2.0;
      const double b = (a1-4.0*a2+3.0*a3)/2.0;
      const double c = a3;
    
      // Computed as an integral from 0.5 to the number of bins of the same size as two ago phase + 0.5
      const double x1 = (double)newNumTimings / (double)oldNumTimings * 3.0 + 0.5;  // should be 3.5 if ratio is same
      const double x2 = 0.5;
      const double expectedTotalTime = a*x1*x1*x1/3.0 + b*x1*x1/2.0 + c*x1 - (a*x2*x2*x2/3.0 + b*x2*x2/2.0 + c*x2);
   
    
      // Measure actual time
      double sum = 0.0;
      for(int i=2; i<newNumTimings; ++i){
	sum += timesNew[i];
      }

      const double avg = sum / ((double)(newNumTimings-2));
      const double actualTotalTime = avg * ((double)newNumTimings); // excluding the load balancing abnormal steps

      const double benefit = expectedTotalTime - actualTotalTime;

      // Determine load balance cost
      const double lbcost = timesNew[0] + timesNew[1] - 2.0*avg;

      const double benefitAfterLB = benefit - lbcost;

    
      // Determine whether LB cost outweights the estimated benefit
      CkPrintf("Quadratic Model: lbcost = %f, expected = %f, actual = %f, x1=%f, a1=%f, a2=%f, a3=%f, b1=%d, b2=%d, a=%f, b=%f, c=%f\n", lbcost, expectedTotalTime, actualTotalTime, x1, a1, a2, a3, b1, b2, a, b, c);
    
    
    
      std::map<std::string, std::pair<int,int> >::const_iterator cpsIter;
      for(cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
	const std::string &name = cpsIter->first;
	const std::pair<int,int> &bounds = cpsIter->second;
	const int lb = bounds.first;
	const int ub = bounds.second;
      
	if(benefitAfterLB > 0){
	  CkPrintf("QuadraticModel: Beneficial LB\n");
	  int newval = newControlPoints[name] / 2;
	  if(newval > lb)
	    newControlPoints[name] = newval;
	  else 
	    newControlPoints[name] = lb;
	} else {
	  CkPrintf("QuadraticModel: Detrimental LB\n");
	  int newval = newControlPoints[name] * 2;
	  if(newval < ub)
	    newControlPoints[name] = newval;
	  else 
	    newControlPoints[name] = ub;
	}
      
      }
    }
    

  }  else if( whichTuningScheme == LDBPeriodOptimal) {
    // Assume this is used in this manner:
    //  1) go to next phase
    //  2) request control point
    //  3) load balancing
    //  4) computation



    instrumentedPhase *prevPhase = previousPhaseData();
    
    const std::vector<double> &times = prevPhase->times;
    const int numTimings = times.size();
    
    if( numTimings > 4){

      const int b1 = 2 + (numTimings-2)/2;
      double s1 = 0;
      double s2 = 0;
    
    
      for(int i=2; i<b1; i++){
	s1 += times[i];
      }
      for(int i=b1; i<numTimings; i++){
	s2 += times[i];
      }
      
    
      const double a1 = s1 / (double)(b1-2);
      const double a2 = s2 / (double)(numTimings-b1);
      const double avg = (a1+a1) / 2.0;

      const double m = (a2-a1)/((double)(numTimings-2)/2.0); // An approximation of the slope of the execution times    

      const double ldbStepsTime = times[0] + times[1];
      const double lbcost = ldbStepsTime - 2.0*avg; // An approximation of the 
      

      int newval = roundDouble(sqrt(2.0*lbcost/m));
      
      // We don't really know what to do if m<=0, so we'll just double the period
      if(m<=0)
	newval = 2*numTimings;     
      
      CkPrintf("Optimal Model (double when negative): lbcost = %f, m = %f, new ldbperiod should be %d\n", lbcost, m, newval);    
    
    
      std::map<std::string, std::pair<int,int> >::const_iterator cpsIter;
      for(cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
	// TODO: lookup only control points that are relevant instead of all of them
	const std::string &name = cpsIter->first;
	const std::pair<int,int> &bounds = cpsIter->second;
	const int lb = bounds.first;
	const int ub = bounds.second;
	
	if(newval < lb){
	  newControlPoints[name] = lb;
	} else if(newval > ub){
	  newControlPoints[name] = ub;
	} else {
	  newControlPoints[name] = newval;
	} 
	
      }
      
      
    }
    
 

  } else if ( whichTuningScheme == UseSteering ) {
	  // -----------------------------------------------------------
	  //  STEERING BASED ON KNOWLEDGE

	  // after 3 phases (and only on even steps), do steering performance. Otherwise, just use previous phase's configuration
	  // plans are only generated after 3 phases

	  instrumentedPhase *twoAgoPhase = twoAgoPhaseData();
	  instrumentedPhase *prevPhase = previousPhaseData();

	  if(phase_id%4 == 0){
		  CkPrintf("Steering based on 2 phases ago:\n");
		  twoAgoPhase->print();
		  CkPrintf("\n");
		  fflush(stdout);

		  std::vector<std::map<std::string,int> > possibleNextStepPlans;


		  // ========================================= Concurrency =============================================
		  // See if idle time is high:
		  {
			  double idleTime = twoAgoPhase->idleTime.avg;
			  CkPrintf("Steering encountered idle time (%f)\n", idleTime);
			  fflush(stdout);
			  if(idleTime > 0.10){
				  CkPrintf("Steering encountered high idle time(%f) > 10%%\n", idleTime);
				  CkPrintf("Steering controlPointSpace.size()=%zu\n", controlPointSpace.size());

				  std::map<std::string, std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > > &possibleCPsToTune = CkpvAccess(cp_effects)["Concurrency"];

				  bool found = false;
				  std::string cpName;
				  std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > *info;
				  std::map<std::string, std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > >::iterator iter;
				  for(iter = possibleCPsToTune.begin(); iter != possibleCPsToTune.end(); iter++){
					  cpName = iter->first;
					  info = &iter->second;

					  // Initialize a new plan based on two phases ago
					  std::map<std::string,int> aNewPlan = twoAgoPhase->controlPoints;

					  CkPrintf("Steering found knob to turn\n");
					  fflush(stdout);

					  if(info->first == ControlPoint::EFF_INC){
						  const int maxValue = controlPointSpace[cpName].second;
						  const int twoAgoValue =  twoAgoPhase->controlPoints[cpName];
						  if(twoAgoValue+1 <= maxValue){
							  aNewPlan[cpName] = twoAgoValue+1; // increase from two phases back
						  }
					  } else {
						  const int minValue = controlPointSpace[cpName].second;
						  const int twoAgoValue =  twoAgoPhase->controlPoints[cpName];
						  if(twoAgoValue-1 >= minValue){
							  aNewPlan[cpName] = twoAgoValue-1; // decrease from two phases back
						  }
					  }

					  possibleNextStepPlans.push_back(aNewPlan);

				  }
			  }
		  }

		  // ========================================= Grain Size =============================================
		  // If the grain size is too small, there may be tons of messages and overhead time associated with scheduling
		  {
			  double overheadTime = twoAgoPhase->overheadTime.avg;
			  CkPrintf("Steering encountered overhead time (%f)\n", overheadTime);
			  fflush(stdout);
			  if(overheadTime > 0.10){
				  CkPrintf("Steering encountered high overhead time(%f) > 10%%\n", overheadTime);
				  CkPrintf("Steering controlPointSpace.size()=%zu\n", controlPointSpace.size());

				  std::map<std::string, std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > > &possibleCPsToTune = CkpvAccess(cp_effects)["GrainSize"];

				  bool found = false;
				  std::string cpName;
				  std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > *info;
				  std::map<std::string, std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > >::iterator iter;
				  for(iter = possibleCPsToTune.begin(); iter != possibleCPsToTune.end(); iter++){
					  cpName = iter->first;
					  info = &iter->second;

					  // Initialize a new plan based on two phases ago
					  std::map<std::string,int> aNewPlan = twoAgoPhase->controlPoints;



					  CkPrintf("Steering found knob to turn\n");
					  fflush(stdout);

					  if(info->first == ControlPoint::EFF_INC){
						  const int maxValue = controlPointSpace[cpName].second;
						  const int twoAgoValue =  twoAgoPhase->controlPoints[cpName];
						  if(twoAgoValue+1 <= maxValue){
							  aNewPlan[cpName] = twoAgoValue+1; // increase from two phases back
						  }
					  } else {
						  const int minValue = controlPointSpace[cpName].second;
						  const int twoAgoValue =  twoAgoPhase->controlPoints[cpName];
						  if(twoAgoValue-1 >= minValue){
							  aNewPlan[cpName] = twoAgoValue-1; // decrease from two phases back
						  }
					  }

					  possibleNextStepPlans.push_back(aNewPlan);

				  }

			  }
		  }
		  // ========================================= GPU Offload =============================================
		  // If the grain size is too small, there may be tons of messages and overhead time associated with scheduling
		  {
			  double idleTime = twoAgoPhase->idleTime.avg;
			  CkPrintf("Steering encountered idle time (%f)\n", idleTime);
			  fflush(stdout);
			  if(idleTime > 0.10){
				  CkPrintf("Steering encountered high idle time(%f) > 10%%\n", idleTime);
				  CkPrintf("Steering controlPointSpace.size()=%zu\n", controlPointSpace.size());

				  std::map<std::string, std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > > &possibleCPsToTune = CkpvAccess(cp_effects)["GPUOffloadedWork"];

				  bool found = false;
				  std::string cpName;
				  std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > *info;
				  std::map<std::string, std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > >::iterator iter;
				  for(iter = possibleCPsToTune.begin(); iter != possibleCPsToTune.end(); iter++){
					  cpName = iter->first;
					  info = &iter->second;

					  // Initialize a new plan based on two phases ago
					  std::map<std::string,int> aNewPlan = twoAgoPhase->controlPoints;


					  CkPrintf("Steering found knob to turn\n");
					  fflush(stdout);

					  if(info->first == ControlPoint::EFF_DEC){
						  const int maxValue = controlPointSpace[cpName].second;
						  const int twoAgoValue =  twoAgoPhase->controlPoints[cpName];
						  if(twoAgoValue+1 <= maxValue){
							  aNewPlan[cpName] = twoAgoValue+1; // increase from two phases back
						  }
					  } else {
						  const int minValue = controlPointSpace[cpName].second;
						  const int twoAgoValue =  twoAgoPhase->controlPoints[cpName];
						  if(twoAgoValue-1 >= minValue){
							  aNewPlan[cpName] = twoAgoValue-1; // decrease from two phases back
						  }
					  }

					  possibleNextStepPlans.push_back(aNewPlan);

				  }

			  }
		  }

		  // ========================================= Done =============================================


		  if(possibleNextStepPlans.size() > 0){
			  newControlPoints = possibleNextStepPlans[0];
		  }


		  CkPrintf("Steering done for this phase\n");
		  fflush(stdout);

	  }  else {
		  // This is not a phase to do steering, so stick with previously used values (one phase ago)
		  CkPrintf("not a phase to do steering, so stick with previously planned values (one phase ago)\n");
		  fflush(stdout);
	  }



  } else if( whichTuningScheme == SimulatedAnnealing ) {

    // -----------------------------------------------------------
    //  SIMULATED ANNEALING
    //  Simulated Annealing style hill climbing method
    //
    //  Find the best search space configuration, and try something
    //  nearby it, with a radius decreasing as phases increase

    CkPrintf("Finding best phase\n");
    instrumentedPhase *bestPhase = allData.findBest();  
    CkPrintf("best found:\n"); 
    bestPhase->print(); 
    CkPrintf("\n"); 
    

    const int convergeByPhase = 100;
    // Determine from 0.0 to 1.0 how far along we are
    const double progress = (double) min(effective_phase, convergeByPhase) / (double)convergeByPhase;

    std::map<std::string, std::pair<int,int> >::const_iterator cpsIter;
    for(cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
      const std::string &name = cpsIter->first;
      const std::pair<int,int> &bounds = cpsIter->second;
      const int minValue = bounds.first;
      const int maxValue = bounds.second;
      
      const int before = bestPhase->controlPoints[name];   
  
      const int range = (int)((maxValue-minValue+1)*(1.0-progress));

      int high = min(before+range, maxValue);
      int low = max(before-range, minValue);
      
      newControlPoints[name] = low;
      if(high-low > 0){
	newControlPoints[name] += randInt(high-low, name.c_str(), phase_id); 
      } 
      
    }

  } else if ( whichTuningScheme == DivideAndConquer ) {

	  // -----------------------------------------------------------
	  //  STEERING FOR Divide & Conquer Programs
	  //  This scheme uses no timing information. It just tries to converge to the point where idle time = overhead time.
	  //  For a Fibonacci example, this appears to be a good heurstic for finding the best performing program.
	  //  The scheme can be applied within a single program tree computation, if the tree is being traversed depth first.

	  // after 3 phases (and only on even steps), do steering performance. Otherwise, just use previous phase's configuration
	  // plans are only generated after 3 phases

	  instrumentedPhase *twoAgoPhase = twoAgoPhaseData();
	  instrumentedPhase *prevPhase = previousPhaseData();

	  if(phase_id%4 == 0){
		  CkPrintf("Divide & Conquer Steering based on 2 phases ago:\n");
		  twoAgoPhase->print();
		  CkPrintf("\n");
		  fflush(stdout);

		  std::vector<std::map<std::string,int> > possibleNextStepPlans;


		  // ========================================= Concurrency =============================================
		  // See if idle time is high:
		  {
			  double idleTime = twoAgoPhase->idleTime.avg;
			  double overheadTime = twoAgoPhase->overheadTime.avg;


			  CkPrintf("Divide & Conquer Steering encountered overhead time (%f) idle time (%f)\n",overheadTime, idleTime);
			  fflush(stdout);
			  if(idleTime+overheadTime > 0.10){
				  CkPrintf("Steering encountered high idle+overheadTime time(%f) > 10%%\n", idleTime+overheadTime);
				  CkPrintf("Steering controlPointSpace.size()=%zu\n", controlPointSpace.size());

				  int direction = -1;
				  if (idleTime>overheadTime){
					  // We need to decrease the grain size, or increase the available concurrency
					  direction = 1;
				  }

				  std::map<std::string, std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > > &possibleCPsToTune = CkpvAccess(cp_effects)["Concurrency"];

				  bool found = false;
				  std::string cpName;
				  std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > *info;
				  std::map<std::string, std::pair<int, std::vector<ControlPoint::ControlPointAssociation> > >::iterator iter;
				  for(iter = possibleCPsToTune.begin(); iter != possibleCPsToTune.end(); iter++){
					  cpName = iter->first;
					  info = &iter->second;
					  
					  // Initialize a new plan based on two phases ago
					  std::map<std::string,int> aNewPlan = twoAgoPhase->controlPoints;

					  CkPrintf("Divide & Conquer Steering found knob to turn\n");
					  fflush(stdout);

					  int adjustByAmount = (int)(myAbs(idleTime-overheadTime)*5.0);
					  
					  if(info->first == ControlPoint::EFF_INC){
					    const int minValue = controlPointSpace[cpName].first;
					    const int maxValue = controlPointSpace[cpName].second;
					    const int twoAgoValue =  twoAgoPhase->controlPoints[cpName];
					    const int newVal = closestInRange(twoAgoValue+adjustByAmount*direction, minValue, maxValue);					  
					    CkAssert(newVal <= maxValue && newVal >= minValue);
					    aNewPlan[cpName] = newVal;
					  } else {
					    const int minValue = controlPointSpace[cpName].first;
					    const int maxValue = controlPointSpace[cpName].second;
					    const int twoAgoValue =  twoAgoPhase->controlPoints[cpName];
					    const int newVal = closestInRange(twoAgoValue-adjustByAmount*direction, minValue, maxValue);
					    CkAssert(newVal <= maxValue && newVal >= minValue);
					    aNewPlan[cpName] = newVal;
					  }
					  
					  possibleNextStepPlans.push_back(aNewPlan);
				  }
			  }
		  }

		  if(possibleNextStepPlans.size() > 0){
		    CkPrintf("Divide & Conquer Steering found %zu possible next phases, using first one\n", possibleNextStepPlans.size());
		    newControlPoints = possibleNextStepPlans[0];
		  } else {
		    CkPrintf("Divide & Conquer Steering found no possible next phases\n");
		  }
	  }

  } else if( whichTuningScheme == Simplex ) {

	  // -----------------------------------------------------------
	  //  Nelder Mead Simplex Algorithm
	  //
	  //  A scheme that takes a simplex (n+1 points) and moves it
	  //  toward the minimum, eventually converging there.

	  s.adapt(controlPointSpace, newControlPoints, phase_id, allData);

  } else if( whichTuningScheme == ExhaustiveSearch ) {
    
    // -----------------------------------------------------------
    // EXHAUSTIVE SEARCH
   
    int numDimensions = controlPointSpace.size();
    CkAssert(numDimensions > 0);
  
    vector<int> lowerBounds(numDimensions);
    vector<int> upperBounds(numDimensions); 
  
    int d=0;
    std::map<std::string, pair<int,int> >::iterator iter;
    for(iter = controlPointSpace.begin(); iter != controlPointSpace.end(); iter++){
      //    CkPrintf("Examining dimension %d\n", d);
      lowerBounds[d] = iter->second.first;
      upperBounds[d] = iter->second.second;
      d++;
    }
   
    // get names for each dimension (control point)
    vector<std::string> names(numDimensions);
    d=0;
    for(std::map<std::string, pair<int,int> >::iterator niter=controlPointSpace.begin(); niter!=controlPointSpace.end(); niter++){
      names[d] = niter->first;
      d++;
    }
  
  
    // Create the first possible configuration
    vector<int> config = lowerBounds;
    config.push_back(0);
  
    // Increment until finding an unused configuration
    allData.cleanupNames(); // put -1 values in for any control points missing
    std::vector<instrumentedPhase*> &phases = allData.phases;     

    while(true){
    
      std::vector<instrumentedPhase*>::iterator piter; 
      bool testedConfiguration = false; 
      for(piter = phases.begin(); piter != phases.end(); piter++){ 
      
	// Test if the configuration matches this phase
	bool match = true;
	for(int j=0;j<numDimensions;j++){
	  match &= (*piter)->controlPoints[names[j]] == config[j];
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

    // results are now in config[i]
    
    for(int i=0; i<numDimensions; i++){
      newControlPoints[names[i]] = config[i];
      CkPrintf("Exhaustive search chose:   %s   -> %d\n", names[i].c_str(), config[i]);
    }


  } else {
    CkAbort("Some Control Point tuning strategy must be enabled.\n");
  }


  const double endGenerateTime = CmiWallTimer();
  
  CkPrintf("Time to generate next control point configuration(s): %f sec\n", (endGenerateTime - startGenerateTime) );
  
}





/// Get control point value from range of integers [lb,ub]
int controlPoint(const char *name, int lb, int ub){
  instrumentedPhase *thisPhaseData = controlPointManagerProxy.ckLocalBranch()->currentPhaseData();
  const int phase_id = controlPointManagerProxy.ckLocalBranch()->phase_id;
  std::map<std::string, pair<int,int> > &controlPointSpace = controlPointManagerProxy.ckLocalBranch()->controlPointSpace;
  int result;

  // if we already have control point values for phase, return them
  if( thisPhaseData->controlPoints.count(std::string(name))>0 && thisPhaseData->controlPoints[std::string(name)]>=0 ){
    CkPrintf("Already have control point values for phase. %s -> %d\n", name, (int)(thisPhaseData->controlPoints[std::string(name)]) );
    return thisPhaseData->controlPoints[std::string(name)];
  }
  

  if( phase_id < 4 || whichTuningScheme == AlwaysDefaults){
    // For the first few phases, just use the lower bound, or the default if one was provided 
    // This ensures that the ranges for all the control points are known before we do anything fancy
    result = lb;

    if(defaultControlPointValues.count(std::string(name)) > 0){
      int v = defaultControlPointValues[std::string(name)];
      CkPrintf("Startup phase using default value of %d for  \"%s\"\n", v, name);   
      result = v;
    }

  } else if(controlPointSpace.count(std::string(name)) == 0){
    // if this is the first time we've seen the CP, then return the lower bound
    result = lb;
    
  }  else {
    // Generate a plan one time for each phase
    controlPointManagerProxy.ckLocalBranch()->generatePlan();
    
    // Use the planned value:
    result = controlPointManagerProxy.ckLocalBranch()->newControlPoints[std::string(name)];
    
  }

  if(!isInRange(result,ub,lb)){
    std::cerr << "control point = " << result << " is out of range: " << lb << " " << ub << std::endl;
    fflush(stdout);
    fflush(stderr);
  }
  CkAssert(isInRange(result,ub,lb));
  thisPhaseData->controlPoints[std::string(name)] = result; // was insert() 

  controlPointSpace.insert(std::make_pair(std::string(name),std::make_pair(lb,ub))); 

  CkPrintf("Control Point \"%s\" for phase %d is: %d\n", name, phase_id, result);
  //  thisPhaseData->print();
  
  return result;
}


FLINKAGE int FTN_NAME(CONTROLPOINT, controlpoint)(CMK_TYPEDEF_INT4 *lb, CMK_TYPEDEF_INT4 *ub){
  CkAssert(CkMyPe() == 0);
  return controlPoint("FortranCP", *lb, *ub);
}




/// Inform the control point framework that a named control point affects the priorities of some array  
// void controlPointPriorityArray(const char *name, CProxy_ArrayBase &arraybase){
//   CkGroupID aid = arraybase.ckGetArrayID();
//   int groupIdx = aid.idx;
//   controlPointManagerProxy.ckLocalBranch()->associatePriorityArray(name, groupIdx);
//   //  CkPrintf("Associating control point \"%s\" with array id=%d\n", name, groupIdx );
// }


// /// Inform the control point framework that a named control point affects the priorities of some entry method  
// void controlPointPriorityEntry(const char *name, int idx){
//   controlPointManagerProxy.ckLocalBranch()->associatePriorityEntry(name, idx);
//   //  CkPrintf("Associating control point \"%s\" with EP id=%d\n", name, idx);
// }





/** Determine the next configuration to try using the Nelder Mead Simplex Algorithm.

    This function decomposes the algorithm into a state machine that allows it to
    evaluate one or more configurations through subsequent clls to this function.
    The state diagram is pictured in the NelderMeadStateDiagram.pdf diagram.

    At one point in the algorithm, n+1 parameter configurations must be evaluated,
    so a list of them will be created and they will be evaluated, one per call.

    Currently there is no stopping criteria, but the simplex ought to contract down
    to a few close configurations, and hence not much change will happen after this 
    point.

 */
void simplexScheme::adapt(std::map<std::string, std::pair<int,int> > & controlPointSpace, std::map<std::string,int> &newControlPoints, const int phase_id, instrumentedData &allData){

	if(useBestKnown){
		CkPrintf("Simplex Tuning: Simplex algorithm is done, using best known phase:\n");
		return;
	}


	if(firstSimplexPhase< 0){
		firstSimplexPhase = allData.phases.size()-1;
		CkPrintf("First simplex phase is %d\n", firstSimplexPhase);
	}

	int n = controlPointSpace.size();

	CkAssert(n>=2);


	if(simplexState == beginning){
		// First we evaluate n+1 random points, then we go to a different state
		if(allData.phases.size() < firstSimplexPhase + n+2	){
			CkPrintf("Simplex Tuning: chose random configuration\n");

			// Choose random values from the middle third of the range in each dimension
			std::map<std::string, std::pair<int,int> >::const_iterator cpsIter;
			for(cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
				const std::string &name = cpsIter->first;
				const std::pair<int,int> &bounds = cpsIter->second;
				const int lb = bounds.first;
				const int ub = bounds.second;
				newControlPoints[name] = lb + randInt(ub-lb+1, name.c_str(), phase_id);
			}
		} else {
			// Set initial simplex:
			for(int i=0; i<n+1; i++){
				simplexIndices.insert(firstSimplexPhase+i);
			}
			CkAssert(simplexIndices.size() == n+1);

			// Transition to reflecting state
			doReflection(controlPointSpace, newControlPoints, phase_id, allData);

		}

	} else if (simplexState == reflecting){
		const double recentPhaseTime = allData.phases[allData.phases.size()-2]->medianTime();
		const double previousWorstPhaseTime = allData.phases[worstPhase]->medianTime();

		// Find the highest time from other points in the simplex
		double highestTimeForOthersInSimplex = 0.0;
		for(std::set<int>::iterator iter = simplexIndices.begin(); iter != simplexIndices.end(); ++iter){
			double t = allData.phases[*iter]->medianTime();
			if(*iter != worstPhase && t > highestTimeForOthersInSimplex) {
				highestTimeForOthersInSimplex = t;
			}
		}

		CkPrintf("After reflecting, the median time for the phase is %f, previous worst phase %d time was %f\n", recentPhaseTime, worstPhase, previousWorstPhaseTime);

		if(recentPhaseTime < highestTimeForOthersInSimplex){
			// if y* < yl,  transition to "expanding"
			doExpansion(controlPointSpace, newControlPoints, phase_id, allData);

		} else if (recentPhaseTime <= highestTimeForOthersInSimplex){
			// else if y* <= yi replace ph with p* and transition to "evaluatingOne"
			CkAssert(simplexIndices.size() == n+1);
			simplexIndices.erase(worstPhase);
			CkAssert(simplexIndices.size() == n);
			simplexIndices.insert(pPhase);
			CkAssert(simplexIndices.size() == n+1);
			// Transition to reflecting state
			doReflection(controlPointSpace, newControlPoints, phase_id, allData);

		} else {
			// if y* > yh
			if(recentPhaseTime <= worstTime){
				// replace Ph with P*
				CkAssert(simplexIndices.size() == n+1);
				simplexIndices.erase(worstPhase);
				CkAssert(simplexIndices.size() == n);
				simplexIndices.insert(pPhase);
				CkAssert(simplexIndices.size() == n+1);
				// Because we later will possibly replace Ph with P**, and we just replaced it with P*, we need to update our Ph reference
				worstPhase = pPhase;
				// Just as a sanity check, make sure we don't use the non-updated values.
				worst.clear();
			}

			// Now, form P** and do contracting phase
			doContraction(controlPointSpace, newControlPoints, phase_id, allData);

		}

	} else if (simplexState == doneExpanding){
		const double recentPhaseTime = allData.phases[allData.phases.size()-2]->medianTime();
		const double previousWorstPhaseTime = allData.phases[worstPhase]->medianTime();
		// A new configuration has been evaluated

		// Check to see if y** < y1
		if(recentPhaseTime < bestTime){
			// replace Ph by P**
			CkAssert(simplexIndices.size() == n+1);
			simplexIndices.erase(worstPhase);
			CkAssert(simplexIndices.size() == n);
			simplexIndices.insert(p2Phase);
			CkAssert(simplexIndices.size() == n+1);
		} else {
			// 	replace Ph by P*
			CkAssert(simplexIndices.size() == n+1);
			simplexIndices.erase(worstPhase);
			CkAssert(simplexIndices.size() == n);
			simplexIndices.insert(pPhase);
			CkAssert(simplexIndices.size() == n+1);
		}

		// Transition to reflecting state
		doReflection(controlPointSpace, newControlPoints, phase_id, allData);

	}  else if (simplexState == contracting){
		const double recentPhaseTime = allData.phases[allData.phases.size()-2]->medianTime();
		const double previousWorstPhaseTime = allData.phases[worstPhase]->medianTime();
		// A new configuration has been evaluated

		// Check to see if y** > yh
		if(recentPhaseTime <= worstTime){
			// replace Ph by P**
			CkPrintf("Replacing phase %d with %d\n", worstPhase, p2Phase);
			CkAssert(simplexIndices.size() == n+1);
			simplexIndices.erase(worstPhase);
			CkAssert(simplexIndices.size() == n);
			simplexIndices.insert(p2Phase);
			CkAssert(simplexIndices.size() == n+1);
			// Transition to reflecting state
			doReflection(controlPointSpace, newControlPoints, phase_id, allData);

		} else {
			// 	conceptually we will replace all Pi by (Pi+Pl)/2, but there is nothing to store this until after we have tried all of them
			simplexState = stillContracting;

			// A set of phases for which (P_i+P_l)/2 ought to be evaluated
			stillMustContractList = simplexIndices;

			CkPrintf("Simplex Tuning: Switched to state: stillContracting\n");
		}

	} else if (simplexState == stillContracting){
		CkPrintf("Simplex Tuning: stillContracting found %zu configurations left to try\n", stillMustContractList.size());

		if(stillMustContractList.size()>0){
			int c = *stillMustContractList.begin();
			stillMustContractList.erase(c);
			CkPrintf("Simplex Tuning: stillContracting evaluating configuration derived from phase %d\n", c);

			std::vector<double> cPhaseConfig = pointCoords(allData, c);

			// Evaluate point P by storing new configuration in newControlPoints, and by transitioning to "reflecting" state
			int v = 0;
			for(std::map<std::string, std::pair<int,int> >::iterator cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
				const std::string &name = cpsIter->first;
				const std::pair<int,int> &bounds = cpsIter->second;
				const int lb = bounds.first;
				const int ub = bounds.second;

				double val = (cPhaseConfig[v] + best[v])/2.0;

				newControlPoints[name] = keepInRange((int)val,lb,ub);
				CkPrintf("Simplex Tuning: v=%d Reflected worst %d %s -> %f (ought to be %f )\n", (int)v, (int)worstPhase, (char*)name.c_str(), (double)newControlPoints[name], (double)P[v]);
				v++;
			}
		} else {
			// We have tried all configurations. We should update the simplex to refer to all the newly tried configurations, and start over
			CkAssert(stillMustContractList.size() == 0);
			simplexIndices.clear();
			CkAssert(simplexIndices.size()==0);
			for(int i=0; i<n+1; i++){
				simplexIndices.insert(allData.phases.size()-2-i);
			}
			CkAssert(simplexIndices.size()==n+1);

			// Transition to reflecting state
			doReflection(controlPointSpace, newControlPoints, phase_id, allData);

		}


	} else if (simplexState == expanding){
		const double recentPhaseTime = allData.phases[allData.phases.size()-2]->medianTime();
		const double previousWorstPhaseTime = allData.phases[worstPhase]->medianTime();
		// A new configuration has been evaluated

		// determine if y** < yl
		if(recentPhaseTime < bestTime){
			// replace Ph by P**
			CkAssert(simplexIndices.size() == n+1);
			simplexIndices.erase(worstPhase);
			CkAssert(simplexIndices.size() == n);
			simplexIndices.insert(p2Phase);
			CkAssert(simplexIndices.size() == n+1);
			// Transition to reflecting state
			doReflection(controlPointSpace, newControlPoints, phase_id, allData);

		} else {
			// else, replace ph with p*
			CkAssert(simplexIndices.size() == n+1);
			simplexIndices.erase(worstPhase);
			CkAssert(simplexIndices.size() == n);
			simplexIndices.insert(pPhase);
			CkAssert(simplexIndices.size() == n+1);
			// Transition to reflecting state
			doReflection(controlPointSpace, newControlPoints, phase_id, allData);
		}


	} else {
		CkAbort("Unknown simplexState");
	}

}



/** Replace the worst point with its reflection across the centroid. */
void simplexScheme::doReflection(std::map<std::string, std::pair<int,int> > & controlPointSpace, std::map<std::string,int> &newControlPoints, const int phase_id, instrumentedData &allData){

	int n = controlPointSpace.size();

	printSimplex(allData);

	computeCentroidBestWorst(controlPointSpace, newControlPoints, phase_id, allData);


	// Quit if the diameter of our simplex is small
	double maxr = 0.0;
	for(int i=0; i<n+1; i++){
		//		Compute r^2 of this simplex point from the centroid
		double r2 = 0.0;
		std::vector<double> p = pointCoords(allData, i);
		for(int d=0; d<p.size(); d++){
			double r1 = (p[d] * centroid[d]);
			r2 += r1*r1;
		}
		if(r2 > maxr)
			maxr = r2;
	}

#if 0
	// At some point we quit this tuning
	if(maxr < 10){
		useBestKnown = true;
		instrumentedPhase *best = allData.findBest();
		CkPrintf("Simplex Tuning: Simplex diameter is small, so switching over to best known phase:\n");

		std::map<std::string, std::pair<int,int> >::const_iterator cpsIter;
		for(cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter) {
			const std::string &name = cpsIter->first;
			newControlPoints[name] =  best->controlPoints[name];
		}
	}
#endif

	// Compute new point P* =(1+alpha)*centroid - alpha(worstPoint)

	pPhase = allData.phases.size()-1;
	P.resize(n);
	for(int i=0; i<n; i++){
		P[i] = (1.0+alpha) * centroid[i] - alpha * worst[i] ;
	}

	for(int i=0; i<P.size(); i++){
		CkPrintf("Simplex Tuning: P dimension %d is %f\n", i, P[i]);
	}


	// Evaluate point P by storing new configuration in newControlPoints, and by transitioning to "reflecting" state
	int v = 0;
	for(std::map<std::string, std::pair<int,int> >::iterator cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
		const std::string &name = cpsIter->first;
		const std::pair<int,int> &bounds = cpsIter->second;
		const int lb = bounds.first;
		const int ub = bounds.second;
		newControlPoints[name] = keepInRange((int)P[v],lb,ub);
		CkPrintf("Simplex Tuning: v=%d Reflected worst %d %s -> %f (ought to be %f )\n", (int)v, (int)worstPhase, (char*)name.c_str(), (double)newControlPoints[name], (double)P[v]);
		v++;
	}


	// Transition to "reflecting" state
	simplexState = reflecting;
	CkPrintf("Simplex Tuning: Switched to state: reflecting\n");

}




/** Replace the newly tested reflection with a further expanded version of itself. */
void simplexScheme::doExpansion(std::map<std::string, std::pair<int,int> > & controlPointSpace, std::map<std::string,int> &newControlPoints, const int phase_id, instrumentedData &allData){
	int n = controlPointSpace.size();
	printSimplex(allData);

	// Note that the original Nelder Mead paper has an error when it displays the equation for P** in figure 1.
	// I believe the equation for P** in the text on page 308 is correct.

	// Compute new point P2 = (1+gamma)*P - gamma(centroid)


	p2Phase = allData.phases.size()-1;
	P2.resize(n);
	for(int i=0; i<n; i++){
		P2[i] = ( (1.0+gamma) * P[i] - gamma * centroid[i] );
	}

	for(int i=0; i<P2.size(); i++){
		CkPrintf("P2 aka P** dimension %d is %f\n", i, P2[i]);
	}


	// Evaluate point P** by storing new configuration in newControlPoints, and by transitioning to "reflecting" state
	int v = 0;
	for(std::map<std::string, std::pair<int,int> >::iterator cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
		const std::string &name = cpsIter->first;
		const std::pair<int,int> &bounds = cpsIter->second;
		const int lb = bounds.first;
		const int ub = bounds.second;
		newControlPoints[name] = keepInRange((int)P2[v],lb,ub);
		CkPrintf("Simplex Tuning: v=%d worstPhase=%d Expanding %s -> %f (ought to be %f )\n", (int)v, (int)worstPhase, (char*)name.c_str(), (double)newControlPoints[name], (double)P[v]);
		v++;
	}


	// Transition to "doneExpanding" state
	simplexState = doneExpanding;
	CkPrintf("Simplex Tuning: Switched to state: doneExpanding\n");

}




/** Replace the newly tested reflection with a further expanded version of itself. */
void simplexScheme::doContraction(std::map<std::string, std::pair<int,int> > & controlPointSpace, std::map<std::string,int> &newControlPoints, const int phase_id, instrumentedData &allData){
	int n = controlPointSpace.size();
	printSimplex(allData);

	// Compute new point P2 = beta*Ph + (1-beta)*centroid


	p2Phase = allData.phases.size()-1;
	P2.resize(n);
	for(int i=0; i<n; i++){
		P2[i] = ( beta*worst[i] + (1.0-beta)*centroid[i] );
	}

	for(int i=0; i<P2.size(); i++){
		CkPrintf("P2 aka P** dimension %d is %f\n", i, P2[i]);
	}


	// Evaluate point P** by storing new configuration in newControlPoints, and by transitioning to "reflecting" state
	int v = 0;
	for(std::map<std::string, std::pair<int,int> >::iterator cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
		const std::string &name = cpsIter->first;
		const std::pair<int,int> &bounds = cpsIter->second;
		const int lb = bounds.first;
		const int ub = bounds.second;
		newControlPoints[name] = keepInRange((int)P2[v],lb,ub);
		CkPrintf("Simplex Tuning: v=%d worstPhase=%d Contracting %s -> %f (ought to be %f )\n", (int)v, (int)worstPhase, (char*)name.c_str(), (double)newControlPoints[name], (double)P[v]);
		v++;
	}
	
	
	// Transition to "contracting" state
	simplexState = contracting;
	CkPrintf("Simplex Tuning: Switched to state: contracting\n");

}




void simplexScheme::computeCentroidBestWorst(std::map<std::string, std::pair<int,int> > & controlPointSpace, std::map<std::string,int> &newControlPoints, const int phase_id, instrumentedData &allData){
	int n = controlPointSpace.size();

	// Find worst performing point in the simplex
	worstPhase = -1;
	worstTime = -1.0;
	bestPhase = 10000000;
	bestTime = 10000000;
	for(std::set<int>::iterator iter = simplexIndices.begin(); iter != simplexIndices.end(); ++iter){
		double t = allData.phases[*iter]->medianTime();
		if(t > worstTime){
			worstTime = t;
			worstPhase = *iter;
		}
		if(t < bestTime){
			bestTime = t;
			bestPhase = *iter;
		}
	}
	CkAssert(worstTime != -1.0 && worstPhase != -1 && bestTime != 10000000 && bestPhase != 10000000);

	best = pointCoords(allData, bestPhase);
	CkAssert(best.size() == n);

	worst = pointCoords(allData, worstPhase);
	CkAssert(worst.size() == n);

	// Calculate centroid of the remaining points in the simplex
	centroid.resize(n);
	for(int i=0; i<n; i++){
		centroid[i] = 0.0;
	}

	int numPts = 0;

	for(std::set<int>::iterator iter = simplexIndices.begin(); iter != simplexIndices.end(); ++iter){
		if(*iter != worstPhase){
			numPts ++;
			// Accumulate into the result vector
			int c = 0;
			for(std::map<std::string,int>::iterator citer = allData.phases[*iter]->controlPoints.begin(); citer != allData.phases[*iter]->controlPoints.end(); ++citer){
				centroid[c] += citer->second;
				c++;
			}

		}
	}

	// Now divide the sums by the number of points.
	for(int v = 0; v<centroid.size(); v++) {
		centroid[v] /= (double)numPts;
	}

	CkAssert(centroid.size() == n);

	for(int i=0; i<centroid.size(); i++){
		CkPrintf("Centroid dimension %d is %f\n", i, centroid[i]);
	}


}



std::vector<double> simplexScheme::pointCoords(instrumentedData &allData, int i){
	std::vector<double> result;
	for(std::map<std::string,int>::iterator citer = allData.phases[i]->controlPoints.begin(); citer != allData.phases[i]->controlPoints.end(); ++citer){
		result.push_back((double)citer->second);
	}
	return result;
}




void ControlPointWriteOutputToDisk(){
	CkAssert(CkMyPe() == 0);
	controlPointManagerProxy.ckLocalBranch()->writeOutputToDisk();
}



/*! @} */

#include "ControlPoints.def.h"

#endif

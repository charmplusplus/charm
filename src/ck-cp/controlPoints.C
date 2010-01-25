#include <charm++.h>
#include "controlPoints.h"
#include "trace-controlPoints.h"
#include "LBDatabase.h"
#include "controlPoints.h"
#include "charm++.h"
#include "trace-projections.h"
#include <pathHistory.h>
#include "cp_effects.h"


//  A framework for tuning "control points" exposed by an application. Tuning decisions are based upon observed performance measurements.
 

/** @defgroup ControlPointFramework Automatic Performance Tuning and Steering Framework  */
/**  @{ */

using namespace std;

#define DEFAULT_CONTROL_POINT_SAMPLE_PERIOD  10000000
#define NUM_SAMPLES_BEFORE_TRANSISTION 5
#define OPTIMIZER_TRANSITION 5


//#undef DEBUGPRINT
//#define DEBUGPRINT 4

static void periodicProcessControlPoints(void* ptr, double currWallTime);


// A pointer to this PE's controlpoint manager Proxy
/* readonly */ CProxy_controlPointManager controlPointManagerProxy;
/* readonly */ int random_seed;
/* readonly */ long controlPointSamplePeriod;
/* readonly */ int whichTuningScheme;
/* readonly */ bool writeDataFileAtShutdown;
/* readonly */ bool loadDataFileAtStartup;
/* readonly */ bool shouldGatherMemoryUsage;
/* readonly */ bool shouldGatherUtilization;
/* readonly */ bool shouldGatherAll;



/// The control point values to be used for the first few phases if the strategy doesn't choose to do something else.
/// These probably come from the command line arguments, so are available only on PE 0
std::map<std::string, int> defaultControlPointValues;



typedef enum tuningSchemeEnum {RandomSelection, SimulatedAnnealing, ExhaustiveSearch, CriticalPathAutoPrioritization, UseBestKnownTiming, UseSteering, MemoryAware}  tuningScheme;



void printTuningScheme(){
  switch(whichTuningScheme){
  case RandomSelection:
    CkPrintf("Tuning Scheme: RandomSelection\n");
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
CkReductionMsg *allMeasuresReduction(int nMsg,CkReductionMsg **msgs){
  double ret[7];
  if(nMsg > 0){
    CkAssert(msgs[0]->getSize()==7*sizeof(double));
    double *m=(double *)msgs[0]->getData();
    ret[0]=m[0];
    ret[1]=m[1];
    ret[2]=m[2];
    ret[3]=m[3];
    ret[4]=m[4];
    ret[5]=m[5];
    ret[6]=m[6];
  }
  for (int i=1;i<nMsg;i++) {
    CkAssert(msgs[i]->getSize()==7*sizeof(double));
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
  }  
  return CkReductionMsg::buildNew(7*sizeof(double),ret);   
}


/// Registers the control point framework's reduction handlers at startup on each PE
/*initproc*/ void registerCPReductions(void) {
  idleTimeReductionType=CkReduction::addReducer(idleTimeReduction);
  allMeasuresReductionType=CkReduction::addReducer(allMeasuresReduction);
}






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



controlPointManager::controlPointManager(){
  generatedPlanForStep = -1;

    exitWhenReady = false;
    alreadyRequestedMemoryUsage = false;   
    alreadyRequestedIdleTime = false;
    alreadyRequestedAll = false;
    
    instrumentedPhase * newPhase = new instrumentedPhase();
    allData.phases.push_back(newPhase);   
    
    dataFilename = (char*)malloc(128);
    sprintf(dataFilename, "controlPointData.txt");
    
    frameworkShouldAdvancePhase = false;
    haveGranularityCallback = false;
//    CkPrintf("[%d] controlPointManager() Constructor Initializing control points, and loading data file\n", CkMyPe());
    
    ControlPoint::initControlPointEffects();

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


  /// Loads the previous run data file
  void controlPointManager::loadDataFile(){
    ifstream infile(dataFilename);
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

      iss >> ips->idleTime.min;
      iss >> ips->idleTime.avg;
      iss >> ips->idleTime.max;

      // Read control point values
      for(int cp=0;cp<numControlPointNames;cp++){
	int cpvalue;
	iss >> cpvalue;
	ips->controlPoints.insert(make_pair(names[cp],cpvalue));
      }

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
    CkPrintf("============= writeDataFile() ============\n");
    ofstream outfile(dataFilename);
    allData.cleanupNames();

    //  string s = allData.toString();
    //  CkPrintf("At end: \n %s\n", s.c_str());

    allData.verify();
    allData.filterOutIncompletePhases();

    outfile << allData.toString();
    outfile.close();
  }

  /// User can register a callback that is called when application should advance to next phase
  void controlPointManager::setCPCallback(CkCallback cb, bool _frameworkShouldAdvancePhase){
    frameworkShouldAdvancePhase = _frameworkShouldAdvancePhase;
    granularityCallback = cb;
    haveGranularityCallback = true;
  }

  /// Called periodically by the runtime to handle the control points
  /// Currently called on each PE
  void controlPointManager::processControlPoints(){

    CkPrintf("[%d] processControlPoints() haveGranularityCallback=%d frameworkShouldAdvancePhase=%d\n", CkMyPe(), (int)haveGranularityCallback, (int)frameworkShouldAdvancePhase);


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
	    
	    if(haveGranularityCallback){ 
#if DEBUGPRINT
	      CkPrintf("Calling granularity change callback\n");
#endif
	      controlPointMsg *msg = new(0) controlPointMsg;
	      granularityCallback.send(msg);
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
    
    if(haveGranularityCallback){ 
      controlPointMsg *msg = new(0) controlPointMsg;
      granularityCallback.send(msg);
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
    
    CkPrintf("gotoNextPhase shouldGatherAll=%d\n", (int)shouldGatherAll);
    fflush(stdout);

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
    




    LBDatabase * myLBdatabase = LBDatabaseObj();

#if CMK_LBDB_ON && 0
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
    

    // Create new entry for the phase we are starting now
    instrumentedPhase * newPhase = new instrumentedPhase();
    allData.phases.push_back(newPhase);
    
    CkPrintf("Now in phase %d allData.phases.size()=%d\n", phase_id, allData.phases.size());

  }

  /// An application uses this to register an instrumented timing for this phase
  void controlPointManager::setTiming(double time){
    currentPhaseData()->times.push_back(time);

#ifdef USE_CRITICAL_PATH_HEADER_ARRAY
       
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
    int size=msg->getSize() / sizeof(double);
    CkAssert(size==3);
    double *r=(double *) msg->getData();
        
    instrumentedPhase* prevPhase = previousPhaseData();
    if(prevPhase != NULL){
      prevPhase->idleTime.min = r[0];
      prevPhase->idleTime.avg = r[1]/CkNumPes();
      prevPhase->idleTime.max = r[2];
      prevPhase->idleTime.print();
      CkPrintf("Stored idle time min=%lf in prevPhase=%p\n", prevPhase->idleTime.min, prevPhase);
    } else {
      CkPrintf("There is no previous phase to store the idle time measurements\n");
    }
    
    alreadyRequestedIdleTime = false;
    checkForShutdown();
    delete msg;
  }






  /// Entry method called on all PEs to request CPU utilization statistics and memory usage
  void controlPointManager::requestAll(CkCallback cb){
    const double i = localControlPointTracingInstance()->idleRatio();
    const double o = localControlPointTracingInstance()->overheadRatio();
    const double m = localControlPointTracingInstance()->memoryUsageMB();
    
    double data[3+3+1];

    double *idle = data;
    double *over = data+3;
    double *mem = data+6;

    idle[0] = i;
    idle[1] = i;
    idle[2] = i;

    over[0] = o;
    over[1] = o;
    over[2] = o;

    mem[0] = m;
    
    localControlPointTracingInstance()->resetAll();

    contribute(7*sizeof(double),data,allMeasuresReductionType, cb);
  }
  
  /// All processors reduce their memory usages in requestIdleTime() to this method
  void controlPointManager::gatherAll(CkReductionMsg *msg){
    int size=msg->getSize() / sizeof(double);
    CkAssert(size==7);
    double *data=(double *) msg->getData();
        
    double *idle = data;
    double *over = data+3;
    double *mem = data+6;

    //    std::string b = allData.toString();

    instrumentedPhase* prevPhase = previousPhaseData();
    if(prevPhase != NULL){
      prevPhase->idleTime.min = idle[0];
      prevPhase->idleTime.avg = idle[1]/CkNumPes();
      prevPhase->idleTime.max = idle[2];
      
      prevPhase->memoryUsageMB = mem[0];
      
      CkPrintf("Stored idle time min=%lf, mem=%lf in prevPhase=%p\n", (double)prevPhase->idleTime.min, (double)prevPhase->memoryUsageMB, prevPhase);

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
       CkPrintf("controlPointManager::exitIfReady exiting immediately\n");
       doExitNow();
     } else {
       CkPrintf("controlPointManager::exitIfReady Delaying exiting\n");
       exitWhenReady = true;
     }
  }



  void controlPointManager::doExitNow(){
    if(writeDataFileAtShutdown){
      CkPrintf("[%d] controlPointShutdown() at CkExit()\n", CkMyPe());
      controlPointManagerProxy.ckLocalBranch()->writeDataFile();
    }
    CkExit();
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


  /// Inform the control point framework that a named control point affects the priorities of some array  
  void controlPointManager::associatePriorityArray(const char *name, int groupIdx){
    CkPrintf("Associating control point \"%s\" affects priority of array id=%d\n", name, groupIdx );
    
    if(affectsPrioritiesArray.count(std::string(name)) > 0 ) {
      affectsPrioritiesArray[std::string(name)].insert(groupIdx);
    } else {
      std::set<int> s;
      s.insert(groupIdx);
      affectsPrioritiesArray[std::string(name)] = s;
    }
    
#if DEBUGPRINT   
    std::map<std::string, std::set<int> >::iterator f;
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
  void controlPointManager::associatePriorityEntry(const char *name, int idx){
    CkPrintf("Associating control point \"%s\" with EP id=%d\n", name, idx);

      if(affectsPrioritiesEP.count(std::string(name)) > 0 ) {
      affectsPrioritiesEP[std::string(name)].insert(idx);
    } else {
      std::set<int> s;
      s.insert(idx);
      affectsPrioritiesEP[std::string(name)] = s;
    }
    
#if DEBUGPRINT
    std::map<std::string, std::set<int> >::iterator f;
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
  


/// An interface callable by the application.
void gotoNextPhase(){
  controlPointManagerProxy.ckLocalBranch()->gotoNextPhase();
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
    
    
    double period;
    bool haveSamplePeriod = CmiGetArgDoubleDesc(args->argv,"+CPSamplePeriod", &period,"The time between Control Point Framework samples (in seconds)");
    if(haveSamplePeriod){
      CkPrintf("LBPERIOD = %ld sec\n", period);
      controlPointSamplePeriod =  period * 1000; /**< A readonly */
    } else {
      controlPointSamplePeriod =  DEFAULT_CONTROL_POINT_SAMPLE_PERIOD;
    }
    
    
    
    whichTuningScheme = RandomSelection;


    if( CmiGetArgFlagDesc(args->argv,"+CPSchemeRandom","Randomly Select Control Point Values") ){
      whichTuningScheme = RandomSelection;
    } else if ( CmiGetArgFlagDesc(args->argv,"+CPExhaustiveSearch","Exhaustive Search of Control Point Values") ){
      whichTuningScheme = ExhaustiveSearch;
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

   loadDataFileAtStartup = false;   
    if( CmiGetArgFlagDesc(args->argv,"+CPLoadData","Load Control Point timings & configurations at startup") ){
      loadDataFileAtStartup = true;
    }


    controlPointManagerProxy = CProxy_controlPointManager::ckNew();
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

/// Shutdown the control point framework, writing data to disk if necessary
extern "C" void controlPointShutdown(){
  if(CkMyPe() == 0){

    // wait for gathering of idle time & memory usage to complete
    controlPointManagerProxy.ckLocalBranch()->exitIfReady();

  }
}

/// A function called at startup on each node to register controlPointShutdown() to be called at CkExit()
void controlPointInitNode(){
//  CkPrintf("controlPointInitNode()\n");
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
/// @note eventually there should be a plugin system where multiple schemes can be plugged in(similar to LB)
void controlPointManager::generatePlan() {
  const int phase_id = this->phase_id;
  const int effective_phase = allData.phases.size();

  // Only generate a plan once per phase
  if(generatedPlanForStep == phase_id) 
    return;
  generatedPlanForStep = phase_id;
 
  CkPrintf("Generating Plan for phase %d\n", phase_id); 
  printTuningScheme();
  
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
 
    if(phase_id%4 == 0){
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
	CkPrintf("Steering (memory based) controlPointSpace.size()=\n", controlPointSpace.size());
	
	// Initialize plan to be the values from two phases ago (later we'll adjust this)
	std::map<std::string, std::pair<int,int> >::const_iterator cpsIter;
	for(cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
	  const std::string &name = cpsIter->first;
	  const int& twoAgoValue =  twoAgoPhase->controlPoints[name];
	  newControlPoints[name] = twoAgoValue;
	}
	CkPrintf("Steering (memory based) initialized plan\n");
	fflush(stdout);

	// look for a possible control point knob to turn
	std::map<std::string, std::vector<std::pair<int, ControlPoint::ControlPointAssociation> > > &possibleCPsToTune = CkpvAccess(cp_effects)["MemoryConsumption"];
	
	// FIXME: assume for now that we just have one control point with the effect, and one direction to turn it
	bool found = false;
	std::string cpName;
	std::vector<std::pair<int, ControlPoint::ControlPointAssociation> > *info;
	std::map<std::string, std::vector<std::pair<int, ControlPoint::ControlPointAssociation> > >::iterator iter;
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
      
      // See if idle time is high:
      double idleTime = twoAgoPhase->idleTime.avg;
      CkPrintf("Steering encountered idle time (%f)\n", idleTime);
      fflush(stdout);
      if(idleTime > 0.10){
	CkPrintf("Steering encountered high idle time(%f) > 10%%\n", idleTime);
	CkPrintf("Steering controlPointSpace.size()=\n", controlPointSpace.size());

	// Initialize plan to be the values from two phases ago (later we'll adjust this)
	std::map<std::string, std::pair<int,int> >::const_iterator cpsIter;
	for(cpsIter=controlPointSpace.begin(); cpsIter != controlPointSpace.end(); ++cpsIter){
	  const std::string &name = cpsIter->first;
	  const int& twoAgoValue =  twoAgoPhase->controlPoints[name];
	  newControlPoints[name] = twoAgoValue;
	}
	CkPrintf("Steering initialized plan\n");
	fflush(stdout);

	// look for a possible control point knob to turn
	std::map<std::string, std::vector<std::pair<int, ControlPoint::ControlPointAssociation> > > &possibleCPsToTune = CkpvAccess(cp_effects)["Concurrency"];
	
	// FIXME: assume for now that we just have one control point with the effect
	bool found = false;
	std::string cpName;
	std::vector<std::pair<int, ControlPoint::ControlPointAssociation> > *info;
	std::map<std::string, std::vector<std::pair<int, ControlPoint::ControlPointAssociation> > >::iterator iter;
	for(iter = possibleCPsToTune.begin(); iter != possibleCPsToTune.end(); iter++){
	  cpName = iter->first;
	  info = &iter->second;
	  found = true;
	  break;
	}

	// Adapt the control point value
	if(found){
	  CkPrintf("Steering found knob to turn\n");
	  fflush(stdout);
	  const int twoAgoValue =  twoAgoPhase->controlPoints[cpName];
	  const int maxValue = controlPointSpace[cpName].second;

	  if(twoAgoValue+1 <= maxValue){
	    newControlPoints[cpName] = twoAgoValue+1; // incrase from two phases back
	  }
	}
	
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
  
      const int range = (maxValue-minValue+1)*(1.0-progress);

      int high = min(before+range, maxValue);
      int low = max(before-range, minValue);
      
      newControlPoints[name] = low;
      if(high-low > 0){
	newControlPoints[name] += randInt(high-low, name.c_str(), phase_id); 
      } 
      
    }

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

}





#define isInRange(v,a,b) ( ((v)<=(a)&&(v)>=(b)) || ((v)<=(b)&&(v)>=(a)) )


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
  

  if( phase_id < 4 ){
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

  CkAssert(isInRange(result,ub,lb));
  thisPhaseData->controlPoints[std::string(name)] = result; // was insert() 

  controlPointSpace.insert(std::make_pair(std::string(name),std::make_pair(lb,ub))); 

  CkPrintf("Control Point \"%s\" for phase %d is: %d\n", name, phase_id, result);
  //  thisPhaseData->print();
  
  return result;
}




/// Inform the control point framework that a named control point affects the priorities of some array  
void controlPointPriorityArray(const char *name, CProxy_ArrayBase &arraybase){
  CkGroupID aid = arraybase.ckGetArrayID();
  int groupIdx = aid.idx;
  controlPointManagerProxy.ckLocalBranch()->associatePriorityArray(name, groupIdx);
  //  CkPrintf("Associating control point \"%s\" with array id=%d\n", name, groupIdx );
}


/// Inform the control point framework that a named control point affects the priorities of some entry method  
void controlPointPriorityEntry(const char *name, int idx){
  controlPointManagerProxy.ckLocalBranch()->associatePriorityEntry(name, idx);
  //  CkPrintf("Associating control point \"%s\" with EP id=%d\n", name, idx);
}




/*! @} */


#include "ControlPoints.def.h"

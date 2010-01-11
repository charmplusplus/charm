#include <charm++.h>
#include "controlPoints.h"
#include "trace-controlPoints.h"
#include "LBDatabase.h"
#include "controlPoints.h"
#include "charm++.h"
#include "trace-projections.h"
#include <pathHistory.h>
#include "cp_effects.h"


/**
 *  \addtogroup ControlPointFramework
 *   @{
 *
 */

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


typedef enum tuningSchemeEnum {RandomSelection, SimulatedAnnealing, ExhaustiveSearch, CriticalPathAutoPrioritization, UseBestKnownTiming, UseSteering}  tuningScheme;


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
/*initproc*/ void registerIdleTimeReduction(void) {
  idleTimeReductionType=CkReduction::addReducer(idleTimeReduction);
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

    exitWhenReady = false;
    newControlPointsAvailable = false;
    alreadyRequestedMemoryUsage = false;   
    alreadyRequestedIdleTime = false;
    
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

      instrumentedPhase ips;

      std::istringstream iss(line);

      // Read memory usage for phase
      iss >> ips.memoryUsageMB;
      //     CkPrintf("Memory usage loaded from file: %d\n", ips.memoryUsageMB);

      iss >> ips.idleTime.min;
      iss >> ips.idleTime.avg;
      iss >> ips.idleTime.max;

      // Read control point values
      for(int cp=0;cp<numControlPointNames;cp++){
	int cpvalue;
	iss >> cpvalue;
	ips.controlPoints.insert(make_pair(names[cp],cpvalue));
      }

      double time;

      while(iss >> time){
	ips.times.push_back(time);
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
    allData.phases.push_back(thisPhaseData);
    allData.cleanupNames();
    outfile << allData.toString();
    outfile.close();
  }

  /// User can register a callback that is called when application should advance to next phase
  void controlPointManager::setGranularityCallback(CkCallback cb, bool _frameworkShouldAdvancePhase){
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
  
  /// The data from the previous phase
  instrumentedPhase * controlPointManager::previousPhaseData(){
    int s = allData.phases.size();
    if(s >= 2 && phase_id > 0) {
      return &(allData.phases[s-2]);
    } else {
      return NULL;
    }
  }
  

  /// Called by either the application or the Control Point Framework to advance to the next phase  
  void controlPointManager::gotoNextPhase(){



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




    
    // increment phase id
    phase_id++;
    
    CkPrintf("Now in phase %d\n", phase_id);
    
    // save a copy of the timing information from this phase
    allData.phases.push_back(thisPhaseData);
    
    // clear the timing information that will be used for the next phase
    thisPhaseData.clear();
    
  }

  /// An application uses this to register an instrumented timing for this phase
  void controlPointManager::setTiming(double time){
    thisPhaseData.times.push_back(time);
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
      CkPrintf("Storing idle time measurements\n");
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



  void controlPointManager::checkForShutdown(){
    if( exitWhenReady && !alreadyRequestedMemoryUsage && !alreadyRequestedIdleTime && CkMyPe()==0){
      doExitNow();
    }
  }


  void controlPointManager::exitIfReady(){
     if( !alreadyRequestedMemoryUsage && !alreadyRequestedIdleTime && CkMyPe()==0){
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
    } 

    shouldGatherMemoryUsage = false;
    if ( CmiGetArgFlagDesc(args->argv,"+CPGatherMemoryUsage","Gather memory usage after each phase") ){
      shouldGatherMemoryUsage = true;
    }

    shouldGatherUtilization = false;
    if ( CmiGetArgFlagDesc(args->argv,"+CPGatherUtilization","Gather utilization & Idle time after each phase") ){
      shouldGatherUtilization = true;
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
void registerGranularityChangeCallback(CkCallback cb, bool frameworkShouldAdvancePhase){
  CkAssert(CkMyPe() == 0);
  CkPrintf("Application has registered a control point change callback\n");
  controlPointManagerProxy.ckLocalBranch()->setGranularityCallback(cb, frameworkShouldAdvancePhase);
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




/// Should an optimizer determine the control point values
bool valueShouldBeProvidedByOptimizer(){
#if 0  
  const int effective_phase = controlPointManagerProxy.ckLocalBranch()->allData.phases.size();
  const int phase_id = controlPointManagerProxy.ckLocalBranch()->phase_id; 
  
  std::map<std::string, pair<int,int> > &controlPointSpace = controlPointManagerProxy.ckLocalBranch()->controlPointSpace; 
  
  double spaceSize = 1.0;
  std::map<std::string, pair<int,int> >::iterator iter;
  for(iter = controlPointSpace.begin(); iter != controlPointSpace.end(); iter++){
    spaceSize *= iter->second.second - iter->second.first + 1;
  }

  //  CkPrintf("Control Point Space:\n\t\tnumber of control points = %d\n\t\tnumber of possible configurations = %.0lf\n", controlPointSpace.size(), spaceSize);

  // return false;
  //  return effective_phase > 1 && phase_id > 1;
  //  return effective_phase >= OPTIMIZER_TRANSITION && phase_id > 3;
#else
  return true;
#endif
}





/// Determine a control point value using some optimization scheme (use max known, simmulated annealling, 
/// user observed characteristic to adapt specific control point values.
/// @note eventually there should be a plugin system where multiple schemes can be plugged in(similar to LB)
int valueProvidedByOptimizer(const char * name, int lb, int ub){
  const int phase_id = controlPointManagerProxy.ckLocalBranch()->phase_id;
  const int effective_phase = controlPointManagerProxy.ckLocalBranch()->allData.phases.size();


  if( whichTuningScheme == RandomSelection){

    int result = lb + randInt(ub-lb+1, name, phase_id);
    CkPrintf("Control Point \"%s\" for phase %d chosen randomly to be: %d\n", name, phase_id, result); 
    return result;

  } else if( whichTuningScheme == CriticalPathAutoPrioritization) {

    // -----------------------------------------------------------
    //  USE CRITICAL PATH TO ADJUST PRIORITIES
    //   
    // This scheme will return the median value for the range 
    // early on, and then will transition over to the new control points
    // determined by the critical path adapting code
    if(controlPointManagerProxy.ckLocalBranch()->newControlPointsAvailable){
      int result = controlPointManagerProxy.ckLocalBranch()->newControlPoints[std::string(name)];
      CkPrintf("valueProvidedByOptimizer(): Control Point \"%s\" for phase %d  from \"newControlPoints\" is: %d\n", name, phase_id, result);
      return result;
    }
  
    std::map<std::string, pair<int,int> > &controlPointSpace = controlPointManagerProxy.ckLocalBranch()->controlPointSpace;  

    if(controlPointSpace.count(std::string(name))>0){
      int minValue =  controlPointSpace[std::string(name)].first;
      int maxValue =  controlPointSpace[std::string(name)].second;
      return (minValue+maxValue)/2;
    }
  
  } else if ( whichTuningScheme == UseBestKnownTiming ) {

    // -----------------------------------------------------------
    //  USE BEST KNOWN TIME  
    static bool firstTime = true;
    if(firstTime){
      firstTime = false;
      CkPrintf("Finding best phase\n");
      instrumentedPhase p = controlPointManagerProxy.ckLocalBranch()->allData.findBest(); 
      CkPrintf("p=:\n");
      p.print();
      CkPrintf("\n");
      controlPointManagerProxy.ckLocalBranch()->best_phase = p;
    } 
    
    instrumentedPhase &p = controlPointManagerProxy.ckLocalBranch()->best_phase;
    int result = p.controlPoints[std::string(name)];
    CkPrintf("valueProvidedByOptimizer(): Control Point \"%s\" for phase %d chosen out of best previous phase to be: %d\n", name, phase_id, result);
    return result;
    
  } else if ( whichTuningScheme == UseSteering ) {
    // -----------------------------------------------------------
    //  STEERING BASED ON KNOWLEDGE
  
    // after 3 iterations, start steering performance

    static int count = 0;
    count++;
    instrumentedPhase *p = controlPointManagerProxy.ckLocalBranch()->previousPhaseData();
    std::map<std::string, pair<int,int> > &controlPointSpace = controlPointManagerProxy.ckLocalBranch()->controlPointSpace;  
    int minValue =  controlPointSpace[std::string(name)].first;
    int maxValue =  controlPointSpace[std::string(name)].second;
    int result = minValue;
 
    if(count > 3){
      CkPrintf("Steering strategy\n");
      CkPrintf("Steering based on previous phase =:\n");
      p->print();
      CkPrintf("\n");
      fflush(stdout);
      
      // See if idle time is high:
      double idleTime = p->idleTime.avg;
      CkPrintf("Steering encountered idle time (%f)\n", idleTime);
      if(idleTime > 0.10){
	CkPrintf("Steering encountered high idle time(%f) > 10%%\n", idleTime);
	
	// look for a possible control point knob to turn
       
	std::map<std::string, std::vector<std::pair<int, ControlPoint::ControlPointAssociation> > > &possibleCPsToTune = CkpvAccess(cp_effects)["Concurrency"];

	// FIXME: assume for now that we just have one control point with the effect
	bool found = false;
	std::string cpName = "";
	std::vector<std::pair<int, ControlPoint::ControlPointAssociation> > *info;
	std::map<std::string, std::vector<std::pair<int, ControlPoint::ControlPointAssociation> > >::iterator iter;
	for(iter = possibleCPsToTune.begin(); iter != possibleCPsToTune.end(); iter++){
	  cpName = iter->first;
	  info = &iter->second;
	  found = true;
	  break;
	}


	result = p->controlPoints[std::string(name)] + 1; // increase it from previous phase

	if(found && result <= maxValue){
	  CkPrintf("valueProvidedByOptimizer(): Steering found a control point to adjust: %s\n", name);

	} else {
	  // Don't have any control points to turn :(
	  CkPrintf("valueProvidedByOptimizer(): Steering didn't find any control points to adjust\n");
	  result = p->controlPoints[std::string(name)];
	}

      }
      
    }
  
    CkPrintf("valueProvidedByOptimizer(): Control Point \"%s\" for phase %d chosen by Steering to be: %d\n", name, phase_id, result);
    return result;
    
  } else if( whichTuningScheme == SimulatedAnnealing ){
    
    // -----------------------------------------------------------
    //  SIMULATED ANNEALING
    //  Simulated Annealing style hill climbing method
    //
    //  Find the best search space configuration, and try something
    //  nearby it, with a radius decreasing as phases increase
  
    std::map<std::string, pair<int,int> > &controlPointSpace = controlPointManagerProxy.ckLocalBranch()->controlPointSpace;  
  
    CkPrintf("Finding best phase\n");
    instrumentedPhase p = controlPointManagerProxy.ckLocalBranch()->allData.findBest();  
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

  } else if( whichTuningScheme == ExhaustiveSearch ){

    // -----------------------------------------------------------
    // EXHAUSTIVE SEARCH

    std::map<std::string, pair<int,int> > &controlPointSpace = controlPointManagerProxy.ckLocalBranch()->controlPointSpace;
    std::set<std::string> &staticControlPoints = controlPointManagerProxy.ckLocalBranch()->staticControlPoints;  
   
    int numDimensions = controlPointSpace.size();
    CkAssert(numDimensions > 0);
  
    vector<int> lowerBounds(numDimensions);
    vector<int> upperBounds(numDimensions); 
  
    int d=0;
    std::map<std::string, pair<int,int> >::iterator iter;
    for(iter = controlPointSpace.begin(); iter != controlPointSpace.end(); iter++){
      //    CkPrintf("Examining dimension %d\n", d);

#if DEBUGPRINT
      std::string name = iter->first;
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
    for(std::map<std::string, pair<int,int> >::iterator niter=controlPointSpace.begin(); niter!=controlPointSpace.end(); niter++){
      s[d] = niter->first;
      // cout << "s[" << d << "]=" << s[d] << endl;
      d++;
    }
  
  
    // Create the first possible configuration
    vector<int> config = lowerBounds;
    config.push_back(0);
  
    // Increment until finding an unused configuration
    controlPointManagerProxy.ckLocalBranch()->allData.cleanupNames(); // put -1 values in for any control points missing
    std::vector<instrumentedPhase> &phases = controlPointManagerProxy.ckLocalBranch()->allData.phases;     

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

  } else {
    CkAbort("Some Control Point tuning strategy must be enabled.\n");
  }

  return 0;  
}





#define isInRange(v,a,b) ( ((v)<=(a)&&(v)>=(b)) || ((v)<=(b)&&(v)>=(a)) )


/// Get control point value from range of integers [lb,ub]
int controlPoint(const char *name, int lb, int ub){
  instrumentedPhase &thisPhaseData = controlPointManagerProxy.ckLocalBranch()->thisPhaseData;
  const int phase_id = controlPointManagerProxy.ckLocalBranch()->phase_id;
  std::map<std::string, pair<int,int> > &controlPointSpace = controlPointManagerProxy.ckLocalBranch()->controlPointSpace;
  int result;

  // if we already have control point values for phase, return them
  if( thisPhaseData.controlPoints.count(std::string(name))>0 ){
    return thisPhaseData.controlPoints[std::string(name)];
  }

  if(controlPointSpace.count(std::string(name)) == 0){
    // if this is the first time we've seen the range for the CP, then return the average
    result = (lb + ub) / 2;
  } else {
    // otherwise, get new values from optimizer
    result = valueProvidedByOptimizer(name, lb, ub);
  }

  CkAssert(isInRange(result,ub,lb));
  thisPhaseData.controlPoints.insert(std::make_pair(std::string(name),result)); 
  controlPointSpace.insert(std::make_pair(std::string(name),std::make_pair(lb,ub))); 

  return result;
}


/// Get control point value from set of provided integers
#if 0
int controlPoint(const char *name, std::vector<int>& values){
  instrumentedPhase &thisPhaseData = controlPointManagerProxy.ckLocalBranch()->thisPhaseData;
  const int phase_id = controlPointManagerProxy.ckLocalBranch()->phase_id;

  int result = valueProvidedByOptimizer(name, 0, values.size() );

  bool found = false;
  for(int i=0;i<values.size();i++){
    if(values[i] == result)
      found = true;
  }
  CkAssert(found);

  thisPhaseData.controlPoints.insert(std::make_pair(std::string(name),result)); 
  return result;
}
#endif



// Dynamic point varies throughout the life of program
// The value it returns is based upon phase_id, a counter that changes for each phase of computation
#if 0
int controlPoint2Pow(const char *name, int fine_granularity, int coarse_granularity){
 instrumentedPhase &thisPhaseData = controlPointManagerProxy.ckLocalBranch()->thisPhaseData;
  const int phase_id = controlPointManagerProxy.ckLocalBranch()->phase_id;

  int result;

  // Use best configuration after a certain point
  if(valueShouldBeProvidedByOptimizer()){
    result = valueProvidedByOptimizer(name);
  } 
  else {

    int l1 = (int)CmiLog2(fine_granularity);
    int l2 = (int)CmiLog2(coarse_granularity);
  
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

  thisPhaseData.controlPoints.insert(std::make_pair(std::string(name),result));

  return result;
}
#endif






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

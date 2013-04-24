/**
   @addtogroup ComlibConverseStrategy

   @{
   @file
   @brief Implementations of the classes in convcomlibstrategy.C
*/

#include "convcomlibmanager.h"

Strategy::Strategy() : PUP::able() {
  ComlibPrintf("Creating new strategy (Strategy constructor)\n");
    myHandle = CkpvAccess(conv_com_object).insertStrategy(this);
    type = CONVERSE_STRATEGY;
    //converseStrategy = this;
    //higherLevel = this;
    isStrategyBracketed = 0;
    //destinationHandler = 0;
}

//Each strategy must define his own Pup interface.
void Strategy::pup(PUP::er &p){ 
    
  //PUP::able::pup(p);

    p | isStrategyBracketed;
    p | type;
    //p | destinationHandler;
    p | myHandle;

    /*if (p.isUnpacking()) {
      converseStrategy = this;
      higherLevel = this;
    }*/
}

//Message holder functions. Message holder is a wrapper around a
//message. Has other useful data like destination processor list for a
//multicast etc.

void MessageHolder::pup(PUP::er &p) {
    //PUP::able::pup(p);

    p | dest_proc;
    p | isDummy;
    p | size;
    p | npes;

    if(p.isUnpacking()) {
        data = (char *)CmiAlloc(size);
        
        if(npes >0)
            pelist = new int[npes];
    }

    p(data, size);
    if(npes > 0)
        p(pelist, npes);    
    else
        pelist = 0;
}

StrategyWrapper::StrategyWrapper(int count) {
  nstrats = count;
  strategy = new Strategy* [nstrats];
  position = new int[nstrats];
  replace = new bool[nstrats];
}

StrategyWrapper::~StrategyWrapper() {
  delete[] strategy;
  delete[] position;
  delete[] replace;
}

void StrategyWrapper::pup (PUP::er &p) {
  p | nstrats;
  //p | total_nstrats;

  if(p.isUnpacking()) {
    strategy = new Strategy* [nstrats];
    position = new int[nstrats];
    replace = new bool[nstrats];
  }
  
  for(int count = 0; count < nstrats; ++count) {
    p | strategy[count];
    p | position[count];
    p | replace[count];
  }
}


StrategyTableEntry::StrategyTableEntry() {
    lastKnownIteration = STARTUP_ITERATION;
    strategy = NULL;
    isNew = 0;
    isReady = 0;
    // WARNING: This constructor is called before CkMyPe() returns the correct results
    errorMode = NORMAL_MODE;
    errorModeServer = NORMAL_MODE_SERVER; 
    discoveryMode = NORMAL_DISCOVERY_MODE;
    bracketedSetupFinished = 0;

    numBufferReleaseReady = 0;

    numElements = 0;
    nBeginItr = 0;
    nEndItr = 0;
    call_doneInserting = 0; 

    // values used during bracketed error/confirm mode
    nEndSaved = 0;
    totalEndCounted = 0;
    nProcSync = 0;

    peConfirmCounter=0;
    total=0; 
        
    // initialize values used during the discovery process
    peList = NULL;
}

//called during learning, when all fields except
//strategy need to be zeroed out
void StrategyTableEntry::reset() {
    numElements = 0;   //used by the array listener, 
                       //could also be used for other objects
    //elementCount = 0;  //Count of how many elements have deposited
                       //their data
    nEndItr = 0;       //#elements that called end iteration
    call_doneInserting = 0;
}


const char *StrategyTableEntry::errorModeString(){
  switch(errorMode) {
  case NORMAL_MODE:
    return "NORMAL_MODE       ";
  case ERROR_MODE:
    return "ERROR_MODE        ";
  case CONFIRM_MODE:
    return "CONFIRM_MODE      ";
  case ERROR_FIXED_MODE:
    return "ERROR_FIXED_MODE  ";
  default:
    return "Unknown Error Mode";
  }
}


const char *StrategyTableEntry::errorModeServerString(){
  if(CkMyPe() == 0){
    switch(errorModeServer) {
    case NORMAL_MODE_SERVER:
      return "NORMAL_MODE_SERVER     ";
    case ERROR_MODE_SERVER:
      return "ERROR_MODE_SERVER      ";
    case CONFIRM_MODE_SERVER:
      return "CONFIRM_MODE_SERVER    ";
    case ERROR_FIXED_MODE_SERVER:
      return "ERROR_FIXED_MODE_SERVER";
    case NON_SERVER_MODE_SERVER:
      return "NON_SERVER_MODE_SERVER ";
    default:
      return "Unknown Server Error Mode";
    }
  } else {
    return "";
  }
}

const char *StrategyTableEntry::discoveryModeString(){
  switch(discoveryMode) {
  case NORMAL_DISCOVERY_MODE:
    return "NORMAL_DISCOVERY_MODE  ";
  case STARTED_DISCOVERY_MODE: 
    return "STARTED_DISCOVERY_MODE ";
  case FINISHED_DISCOVERY_MODE:
    return "FINISHED_DISCOVERY_MODE";
  default:
    return "Unknown Discovery Mode ";
  }
}






//PUPable_def(Strategy);
PUPable_def(MessageHolder)

//CsvDeclare(int, pipeBcastPropagateHandle);
//CsvDeclare(int, pipeBcastPropagateHandle_frag);

/*@}*/

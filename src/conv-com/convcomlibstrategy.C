
#include <converse.h>
#include "convcomlibstrategy.h"

//Class that defines the entry methods that a strategy must define.
//To write a new strategy inherit from this class and define the
//virtual methods.  Every strategy can also define its own constructor
//and have any number of arguments. Also call the parent class methods
//in those methods. The strategies defined here can only be used in
//converse. For Charm++ strategies please see ck-core/ComlibStraetgy.h

Strategy::Strategy() : PUP::able() {
    myInstanceID = 0;
    type = CONVERSE_STRATEGY;
    converseStrategy = this;
    isStrategyBracketed = 0;
};

//Each strategy must define his own Pup interface.
void Strategy::pup(PUP::er &p){ 

    PUP::able::pup(p);

    p | isStrategyBracketed;
    p | type;
}


//Message holder functions. Message holder is a wrapper around a
//message. Has other useful data like destination processor list for a
//multicast etc.

MessageHolder::MessageHolder(char * msg, int proc, int sz) : PUP::able() {
    data = msg;
    dest_proc = proc;
    size = sz;
    
    isDummy = 0;

    npes = 0;
    pelist = 0;
}

MessageHolder::~MessageHolder() {
    /*
      if(pelist != NULL && npes > 0)
      delete[] pelist;
    */
}

char * MessageHolder::getMessage(){
    return data;
}

void MessageHolder::pup(PUP::er &p) {
    PUP::able::pup(p);

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

void StrategyWrapper::pup (PUP::er &p) {

    //CkPrintf("In PUP of StrategyWrapper\n");

    p | nstrats;
    if(p.isUnpacking())
	s_table = new Strategy * [nstrats];
    
    for(int count = 0; count < nstrats; count ++)
        p | s_table[count];
}


StrategyTableEntry::StrategyTableEntry() {
    strategy = NULL;
    
    numElements = 0;   //used by the array listener, 
                       //could also be used for other objects
    elementCount = 0;  //Count of how many elements have deposited
                       //their data
    nEndItr = 0;       //#elements that called end iteration
    call_doneInserting = 0;
}


PUPable_def(Strategy);
PUPable_def(MessageHolder);

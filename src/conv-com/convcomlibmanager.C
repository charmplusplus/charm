/* Converse ComlibManager 
   Enables communication library strategies to be called from converse code.
   Also called by the Charm Comlibmanager.
   
   Stores a strategy table. Strategies can be inserted and accessed
   from this table.

   Sameer Kumar 28/03/04
*/

#include "convcomlibmanager.h"
#include "routerstrategy.h"

int comm_debug;

CkpvDeclare(ConvComlibManager *, conv_comm_ptr);
CkpvDeclare(int, RecvdummyHandle);


ConvComlibManager::ConvComlibManager(): strategyTable(10){
    nstrats = 0;
}

void ConvComlibManager::insertStrategy(Strategy *s) {
    StrategyTableEntry &st = strategyTable[nstrats];
    st.strategy = s;

    s->setInstance(nstrats);
    // if the strategy is pure converse or pure charm the following line is a
    // duplication, but if a charm strategy embed a converse strategy it is
    // necessary to set the instanceID in both
    s->getConverseStrategy()->setInstance(nstrats);
    nstrats ++;
}


void ConvComlibManager::insertStrategy(Strategy *s, int loc) {

    //For now allow insertion of any location    
    StrategyTableEntry &st = strategyTable[loc];

    st.strategy = s;
}

//handler for dummy messages
void recv_dummy(void *msg){
    ComlibPrintf("Received Dummy %d\n", CkMyPe());    
    CmiFree(msg);
}

extern void propagate_handler(void *);
extern void propagate_handler_frag(void *);

//An initialization routine which does preliminary initialization of the 
//Converse commlib manager. Currently also initialized krishnans code
void initComlibManager(){ 
    CkpvInitialize(ConvComlibManager *, conv_comm_ptr);
    ConvComlibManager *conv_com = new ConvComlibManager();
    CkpvAccess(conv_comm_ptr) = conv_com;
    
    //comm_debug = 1;
    ComlibPrintf("Init Call\n");
    
    CkpvInitialize(int, RecvdummyHandle);
    CkpvAccess(RecvdummyHandle) = CkRegisterHandler((CmiHandler)recv_dummy);

    // init strategy specific variables
    CsvInitialize(int, pipeBcastPropagateHandle);
    CsvInitialize(int, pipeBcastPropagateHandle_frag);
    CsvAccess(pipeBcastPropagateHandle) = CmiRegisterHandler((CmiHandler)propagate_handler);
    CsvAccess(pipeBcastPropagateHandle_frag) = CmiRegisterHandler((CmiHandler)propagate_handler_frag);

    PUPable_reg(Strategy);
    PUPable_reg(RouterStrategy);
    PUPable_reg(MessageHolder);
}
 
Strategy *ConvComlibGetStrategy(int loc) {
    //Calling converse strategy lets Charm++ strategies one strategy
    //table entry but multiple layers of strategies (Charm on top of Converse).
    return (CkpvAccess(conv_comm_ptr))->getStrategy(loc)->getConverseStrategy();
}

void ConvComlibRegisterStrategy(Strategy *s) {
    (CkpvAccess(conv_comm_ptr))->insertStrategy(s);    
}

void ConvComlibScheduleDoneInserting(int loc) {
    (* (CkpvAccess(conv_comm_ptr))->getStrategyTable())[loc].
        call_doneInserting++;
}

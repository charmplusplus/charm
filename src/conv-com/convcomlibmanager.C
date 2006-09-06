
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

CkpvDeclare(ConvComlibManager, conv_com_object);
CkpvDeclare(ConvComlibManager *, conv_com_ptr);
CkpvDeclare(int, RecvdummyHandle);

CkpvDeclare(int, strategy_handlerid);

void *strategyHandler(void *msg) {
    CmiMsgHeaderBasic *conv_header = (CmiMsgHeaderBasic *) msg;
    int instid = conv_header->stratid;
    
    Strategy *strat = ConvComlibGetStrategy(instid);
    
    strat->handleMessage(msg);
    return NULL;
}

ConvComlibManager::ConvComlibManager(): strategyTable(MAX_NUM_STRATS){
    nstrats = 0;
    init_flag = CmiFalse;
}

void ConvComlibManager::insertStrategy(Strategy *s) {

    if(nstrats >= MAX_NUM_STRATS)
        CkAbort("Too Many strategies\n");
    
    StrategyTableEntry &st = strategyTable[nstrats];
    
    if(st.strategy != NULL)
        delete st.strategy;

    st.strategy = s;

    s->setInstance(nstrats);
    
    // if the strategy is pure converse or pure charm the following
    // line is a duplication, but if a charm strategy embed a converse
    // strategy it is necessary to set the instanceID in both
    s->getConverseStrategy()->setInstance(nstrats);
    nstrats ++;
}


void ConvComlibManager::insertStrategy(Strategy *s, int loc) {

    if(loc >= MAX_NUM_STRATS)
        CkAbort("Too Many strategies\n");

    //For now allow insertion of any location    
    StrategyTableEntry &st = strategyTable[loc];

    //Check to check for the case where the old strategy is not re inserted 
    if(st.strategy != NULL && st.strategy != s)
        delete st.strategy;
    
    st.strategy = s;
}

//handler for dummy messages
void recv_dummy(void *msg){
    ComlibPrintf("Received Dummy %d\n", CkMyPe());    
    CmiFree(msg);
}

//extern void propagate_handler(void *);
extern void propagate_handler_frag(void *);

//An initialization routine which does prelimnary initialization of the 
//Converse commlib manager. 
void initComlibManager(){ 

    if(!CkpvInitialized(conv_com_object))
	CkpvInitialize(ConvComlibManager, conv_com_object);
    
    if(!CkpvInitialized(conv_com_ptr))
	CkpvInitialize(ConvComlibManager *, conv_com_ptr);
    
    if(CkpvAccess(conv_com_object).getInitialized()) 
      return;
    
    CkpvAccess(conv_com_ptr) = &(CkpvAccess(conv_com_object));
    //comm_debug = 1;
    ComlibPrintf("Init Call\n");
    
    CkpvInitialize(int, RecvdummyHandle);
    CkpvAccess(RecvdummyHandle) = CkRegisterHandler((CmiHandler)recv_dummy);

    // init strategy specific variables
    CsvInitialize(int, pipeBcastPropagateHandle);
    CsvInitialize(int, pipeBcastPropagateHandle_frag);
    //CsvAccess(pipeBcastPropagateHandle) = CmiRegisterHandler((CmiHandler)propagate_handler);
    
    CsvAccess(pipeBcastPropagateHandle_frag) = CkRegisterHandler((CmiHandler)propagate_handler_frag);
    
    CkpvInitialize(int, strategy_handlerid);

    CkpvAccess(strategy_handlerid) = CkRegisterHandler((CmiHandler) strategyHandler);

    if (CkMyRank() == 0) {
    PUPable_reg(Strategy);
    PUPable_reg(RouterStrategy);
    PUPable_reg(MessageHolder);
    }
    CkpvAccess(conv_com_object).setInitialized();
}

Strategy *ConvComlibGetStrategy(int loc) {
    //Calling converse strategy lets Charm++ strategies one strategy
    //table entry but multiple layers of strategies (Charm on top of Converse).
    return (CkpvAccess(conv_com_ptr))->getStrategy(loc)->getConverseStrategy();
}

void ConvComlibRegisterStrategy(Strategy *s) {
    (CkpvAccess(conv_com_ptr))->insertStrategy(s);    
}

void ConvComlibScheduleDoneInserting(int loc) {
    (* (CkpvAccess(conv_com_ptr))->getStrategyTable())[loc].
        call_doneInserting++;
}

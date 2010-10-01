#include "reductionBenchmark.h"
#include "myChareArray.h"
#include "testController.h"

/// Converse broadcast and reduction handler function IDs. Global vars.
int bcastHandlerID, rednHandlerID, bcastConverterID;
/// Array elements local to each pe. Used by the conv to array bcast converter
std::vector<MyChareArray*> localElems;
/// Pointer to the mainchare on pe 0 used by the converse redn handler
TestController *mainChare;



/// Converse reduction merge function triggered at each vertex along the reduction spanning tree
void* convRedn_sum (int *size, void *local, void **remote, int count)
{
    CkReductionMsg *localMsg = CkReductionMsg::unpack( EnvToUsr( (envelope*)local ) );
    double *dataBuf          = reinterpret_cast<double*>( localMsg->getData() );
    int msgSize              = localMsg->getSize()/sizeof(double);
    // Reduce all the remote msgs from children in the tree
    for (int i=0; i < count; i++)
    {
        envelope *aEnv       = (envelope*)( remote[i] );
        CkReductionMsg *aMsg = CkReductionMsg::unpack( EnvToUsr(aEnv) );
        CkAssert( localMsg->getSize() == aMsg->getSize() );
        double *anotherBuf   = reinterpret_cast<double*>( aMsg->getData() );
        for (int j=0; j< msgSize; j++)
            dataBuf[j] += anotherBuf[j];
    }
    // Repack the local msg
    CkReductionMsg::pack(localMsg);
    // Return a handle to the final reduced msg
    return local;
}



// Converse Reduction msg handler triggered at the root of the converse reduction
void convRednHandler(void *env)
{
    CkReductionMsg *msg = CkReductionMsg::unpack( EnvToUsr((envelope*)env) );
    #ifdef VERBOSE_OPERATION
        CkPrintf("\n[%d] Converse reduction handler triggered",CkMyPe());
    #endif
    mainChare->receiveReduction(msg);
}



/// Converse broadcast handler
void convBcastHandler(void *env)
{
    #ifdef VERBOSE_OPERATION
        CkPrintf("\n[%d] Received converse bcast",CkMyPe());
    #endif
    /// Touch the data cursorily
    DataMsg *msg = DataMsg::unpack( EnvToUsr((envelope*)env) );
    msg->data[0] = 0;
    /// Prepare some data to be returned
    double *returnData = msg->data;
    CkReductionMsg *redMsg = CkReductionMsg::buildNew( msg->size*sizeof(double), returnData, CkReduction::sum_double);
    CkReductionMsg::pack( redMsg );
    envelope *redEnv = UsrToEnv(redMsg);
    /// Contribute to reduction
    #ifdef VERBOSE_OPERATION
        CkPrintf("\n[%d] Going to trigger reduction using mechanism: %s",CkMyPe(), commName[msg->commType]);
    #endif
    CmiSetHandler(redEnv, rednHandlerID);
    CmiReduce(redEnv, redEnv->getTotalsize(), convRedn_sum);
    /// Delete the incoming msg
    delete msg;
}



/// Converse message handler that translates a converse broadcast to a charm array bcast
void convBcastToArrayBcastHandler(void *env)
{
    #ifdef VERBOSE_OPERATION
        CkPrintf("\n[%d] Received converse bcast. Gonna deliver to local array elements",CkMyPe());
    #endif
    /// Get an unpacked charm msg from the envelope
    DataMsg *msg = DataMsg::unpack( EnvToUsr((envelope*)env) );
    /// Deliver to each local element
    int numElems = localElems.size();
    for (int i=0; i < localElems.size(); i++)
    {
        DataMsg *aMsgPtr = (numElems - i > 1) ? (DataMsg*)CkCopyMsg((void**)&msg) : msg;
        localElems[i]->crunchData(aMsgPtr);
    }
    /// If there are no local elements, delete the incoming msg
    if (numElems == 0)
        delete msg;
}



// Register the converse msg handlers used for the converse bcast/redn test
void registerHandlers()
{
    bcastHandlerID   = CmiRegisterHandler(convBcastHandler);
    rednHandlerID    = CmiRegisterHandler(convRednHandler);
    bcastConverterID = CmiRegisterHandler(convBcastToArrayBcastHandler);
}

#include "reductionBenchmark.def.h"


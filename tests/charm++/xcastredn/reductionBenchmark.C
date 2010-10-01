#include "reductionBenchmark.h"
#include <iomanip>
#include <math.h>

//--------------- Functions for the converse bcast/redn portion of the test ---------------

/// Readonly proxy to the QHogger group
CProxy_QHogger hogger;
/// Pointer to the mainchare on pe 0 used by the converse redn handler
TestController *mainChare;
/// A list of array elements local to each address space (pe)
std::vector<MyChareArray*> localElems;
/// Converse broadcast and reduction handler function IDs. Global vars.
int bcastHandlerID, rednHandlerID, bcastConverterID;



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



MyChareArray::MyChareArray(CkGroupID grpID): msgNum(0), mcastGrpID(grpID)
{
    #ifdef VERBOSE_CREATION
        CkPrintf("\nArrayElem[%d] Just born...",thisIndex);
    #endif
    mcastMgr = CProxy_CkMulticastMgr(mcastGrpID).ckLocalBranch();
    localElems.push_back(this);
}
//-----------------------------------------------------------------------------------------



inline void MyChareArray::crunchData(DataMsg *msg)
{
    #ifdef VERBOSE_OPERATION
        CkPrintf("\nArrayElem[%d] Received msg number %d", thisIndex, msgNum++);
    #endif
    /// Touch the data cursorily
    msg->data[0] = 0;
    /// Prepare some data to be returned
    double *returnData = msg->data;
    /// Contribute to reduction
    #ifdef VERBOSE_OPERATION
        CkPrintf("\nArrayElem[%d] Going to trigger reduction using mechanism: %s",thisIndex,commName[msg->commType]);
    #endif
    switch (msg->commType)
    {
        case CharmBcast:
            contribute(msg->size*sizeof(double),returnData,CkReduction::sum_double);
            break;

        case CkMulticast:
            CkGetSectionInfo(sid, msg);
            mcastMgr->contribute(msg->size*sizeof(double),returnData,CkReduction::sum_double,sid);
            break;

        case Comlib:
            CkGetSectionInfo(sid, msg);
            mcastMgr->contribute(msg->size*sizeof(double),returnData,CkReduction::sum_double,sid);
            break;

        case ConverseBcast:
            CkAbort("The Converse bcast/redn loop should NOT end up in a chare array entry method! Kick the test writer!");
            break;

        case ConverseToArrayBcast:
            contribute(msg->size*sizeof(double),returnData,CkReduction::sum_double);
            break;

        default:
            CkAbort("Attempting to use unknown mechanism to handle the reduction");
            break;
    }
    delete msg;
}



/// Function that fills the scheduler q and then triggers the test
void qFiller(void *param, void *msg)
{
    #ifdef VERBOSE_OPERATION
        CkPrintf("\n[%d] Filling scheduler Q with %d entry methods of duration %d us", CkMyPe(), cfg.qLength, cfg.uSecs);
    #endif
    /// Build up a scheduler queue of the required length
    for (int i=0; i < cfg.qLength; i++)
        hogger[CkMyPe()].doSomething(cfg.uSecs);
    /// Trigger the test via the callback
    CkCallback *triggerCB =  (CkCallback*)param;
    if (!triggerCB)    CkAbort("Test trigger callback not supplied!");
    #ifdef VERBOSE_OPERATION
        CkPrintf("\n[%d] Triggering test", CkMyPe());
    #endif
    if (CkMyPe() == 0) triggerCB->send();
    delete triggerCB;
}



TestController::TestController(CkArgMsg *m)
{
    /// Set default configs
    cfg.setDefaults();

    //Process command-line arguments
    if (m->argc == 1)
    {
        // just proceed silently for empty args
    }
    else if ( (m->argc >= 2) && (m->argc <= 9) )
    {
        if (m->argc >= 2)
            cfg.numRepeats       = atoi(m->argv[1]);
        if (m->argc >= 3)
            cfg.msgSizeMin       = atoi(m->argv[2]);
        if (m->argc >= 4)
            cfg.msgSizeMax       = atoi(m->argv[3]);
        if (m->argc >= 5)
            cfg.useContiguousSection =(atoi(m->argv[4]) == 0)? false: true;
        if (m->argc >= 6)
            cfg.qLength          = atoi(m->argv[5]);
        if (m->argc >= 7)
            cfg.uSecs            = atoi(m->argv[6]);
        if (m->argc >= 8)
        {
            cfg.sectionSize      = atoi(m->argv[7]);
            if (cfg.arraySize < cfg.sectionSize)
                cfg.arraySize    = cfg.sectionSize;
        }
        if (m->argc >= 9)
        {
            cfg.arraySize        = atoi(m->argv[8]);
            if (cfg.arraySize < cfg.sectionSize)
                CkAbort("Invalid input. Ensure that section size <= array size");
        }
    }
    else
        CkPrintf("Wrong number of arguments. Try %s numRepeats msgSizeMin(bytes) msgSizeMax(KB) isSectionContiguous qFillLength fillMethodDuration(us) sectionDimX sectionDimY sectionDimZ arrayDimX arrayDimY arrayDimZ",m->argv[0]);

    delete m;
    CkPrintf("\nMeasuring performance of chare array collectives using different communication libraries in charm++. \nNum PEs: %d \nInputs are: \n\tArray size: %d \n\tSection size: %d \n\tMsg sizes: %d bytes to %d KB \n\tNum repeats: %d \n\tScheduler Q Fill Length: %d entry methods \n\tScheduler Q Fill Method Duration: %d us",
             CkNumPes(), cfg.arraySize, cfg.sectionSize, cfg.msgSizeMin, cfg.msgSizeMax, cfg.numRepeats, cfg.qLength, cfg.uSecs);

    // Initialize the mainchare pointer used by the converse redn handler
    mainChare = this;
    // Set up a QHogger group to keep the scheduler Q non-empty
    hogger = CProxy_QHogger::ckNew();

    // Setup the multicast manager stuff
    CkGroupID mcastGrpID  = CProxy_CkMulticastMgr::ckNew(4);
    CkMulticastMgr *mgr   = CProxy_CkMulticastMgr(mcastGrpID).ckLocalBranch();

    /// Create the array
    chareArray            = CProxy_MyChareArray::ckNew(mcastGrpID,cfg.arraySize);
    /// Create the array section to use with CkMulticast
    arraySections.push_back( createSection(cfg.useContiguousSection) );

    /// Delegate the section collectives to the multicast manager
    arraySections[0].ckSectionDelegate(mgr);
    /// Setup the client at the root of the reductions
    CkCallback *cb = new CkCallback(CkIndex_TestController::receiveReduction(0),thisProxy);
    chareArray.ckSetReductionClient(cb);
    mgr->setReductionClient(arraySections[0],cb);

    /// Start off with the first comm type and the smallest message size
    curCommType    = CkMulticast;
    curMsgSize     = cfg.msgSizeMin;
    curRepeatNum   = 0;

    /// Prepare the output and logging buffers
    log<<std::fixed<<std::setprecision(6);
    log<<"\n"<<std::setw(cfg.fieldWidth)<<"Msg size (KB)"
             <<std::setw(cfg.fieldWidth)<<"Avg time (ms)"
             <<std::setw(cfg.fieldWidth)<<"Min time (ms)"
             <<std::setw(cfg.fieldWidth)<<"Max time (ms)"
             <<std::setw(cfg.fieldWidth)<<"Std Dev  (ms)";

    out<<std::fixed<<std::setprecision(6);
    out<<"\n\nSummary: Avg time taken (ms) for different msg sizes by each comm mechanism\n"<<std::setw(commNameLen)<<"Mechanism";
    for (int i=cfg.msgSizeMin; i<= cfg.msgSizeMax*1024; i*=2)
        out<<std::setw(cfg.fieldWidth-3)<<(float)i/1024<<std::setw(3)<<" KB";
    out<<"\n"<<std::setw(commNameLen)<<commName[curCommType];

    /// Wait for quiescence and then start the timing tests
    CkCallback *trigger = new CkCallback(CkIndex_TestController::startTest(), thisProxy);
    CkCallback filler(qFiller, (void*)trigger);
    CkStartQD(filler);
}




CProxySection_MyChareArray TestController::createSection(const bool isSectionContiguous)
{
    /// Determine the lower starting index of the section along each dimension
    int Xl = 0;
    /// Determine a step size based on whether a contiguous section is needed
    int dX = 1;
    if (!isSectionContiguous)
        dX = floor( cfg.arraySize / cfg.sectionSize );
    /// Determine the upper index bounds for the section
    int Xu = (cfg.sectionSize - 1) * dX;
    CkAssert(cfg.arraySize >= Xu);
    /// Create the section
    return CProxySection_MyChareArray::ckNew(chareArray,Xl,Xu,dX);
}




void TestController::sendMulticast(const CommMechanism commType, const int msgSize)
{
    #ifdef VERBOSE_STATUS
        CkPrintf("\nMsgSize: %f Sending out multicast number %d",(float)curMsgSize/1024,curRepeatNum+1);
    #endif
    /// Create a message of required size
    int numUnits = curMsgSize/sizeof(double);
    DataMsg *msg = new (numUnits) DataMsg(numUnits,commType);
    /// Fill it with data
    for (int i=0; i<numUnits; i++)
        msg->data[i] = i;
    /// Start the timer and trigger the send to the array / section
    switch (commType)
    {
        case CharmBcast:
            timeStart = CmiWallTimer();
            chareArray.crunchData(msg);
            break;

        case CkMulticast:
            timeStart = CmiWallTimer();
            arraySections[0].crunchData(msg);
            break;

        case Comlib:
            timeStart = CmiWallTimer();
            arraySections[0].crunchData(msg);
            break;

        case ConverseBcast:
        {
            DataMsg::pack(msg);
            envelope *env = UsrToEnv(msg);
            CmiSetHandler(env, bcastHandlerID);
            timeStart = CmiWallTimer();
            CmiSyncBroadcastAllAndFree(env->getTotalsize(), (char*)env);
            break;
        }

        case ConverseToArrayBcast:
        {
            DataMsg::pack(msg);
            envelope *env = UsrToEnv(msg);
            CmiSetHandler(env, bcastConverterID);
            timeStart = CmiWallTimer();
            CmiSyncBroadcastAllAndFree(env->getTotalsize(), (char*)env);
            break;
        }

        default:
            CkAbort("Attempting to use unknown mechanism to communicate with chare array");
            break;
    }
}




void TestController::receiveReduction(CkReductionMsg *msg)
{
    /// Compute the time taken (in milliseconds) for this multicast/reduction loop
    loopTimes.push_back( 1000*(CmiWallTimer() - timeStart) );

    #ifdef VERBOSE_STATUS
        CkPrintf("\nMsgSize: %f Received reduction number %d for repeat number %d", (float)curMsgSize/1024, msg->getRedNo(), curRepeatNum+1);
    #endif

    /// If this is the first ever multicast/reduction loop, dont time it as it includes tree setup times etc
    if (curCommType == CkMulticast && msg->getRedNo() == 0)
    {
        CkPrintf("\nFirst xcast/redn loop took: %.6f ms. Discarding this from collected measurements as it might include tree setup times etc",loopTimes[0]);
        loopTimes.pop_back();
        curRepeatNum--;
    }

    /// If this ends the timings for a msg size
    if (++curRepeatNum >= cfg.numRepeats)
    {
        /// Compute some statistics (avg, std dev, min and max times)
        double avgTime = 0, stdDev = 0;
        double minTime = loopTimes[0], maxTime = loopTimes[0];

        for (int i=0; i< loopTimes.size(); i++)
        {
            avgTime += loopTimes[i];
            stdDev  += loopTimes[i] * loopTimes[i];
            if (loopTimes[i] > maxTime)
                maxTime = loopTimes[i];
            if (loopTimes[i] < minTime)
                minTime = loopTimes[i];
        }
        avgTime /= loopTimes.size();
        stdDev   = sqrt( stdDev/loopTimes.size() - avgTime*avgTime );

        /// Collate the results
        log<<"\n"<<std::setw(cfg.fieldWidth)<<(float)curMsgSize/1024
                 <<std::setw(cfg.fieldWidth)<<avgTime
                 <<std::setw(cfg.fieldWidth)<<minTime
                 <<std::setw(cfg.fieldWidth)<<maxTime
                 <<std::setw(cfg.fieldWidth)<<stdDev;
        out<<std::setw(cfg.fieldWidth)<<avgTime;

        /// Reset counters, timings and sizes
        curRepeatNum = 0;
        curMsgSize *= 2;
        loopTimes.clear();

        /// If this ends the timings for all msg sizes, print results and proceed to the next phase
        if (curMsgSize > cfg.msgSizeMax*1024)
        {
            /// Print the results
            CkPrintf("\n----------------------------------------------------------------");
            CkPrintf("\nFinished timing the collectives mechanism: %s. Results: \n%s\n",commName[curCommType],log.str().c_str());
            /// Clear the output buffer
            log.str("");
            /// Reset the counters
            curMsgSize = cfg.msgSizeMin;
            /// Exit if done
            if (++curCommType >= Comlib)
            {
                CkPrintf("\n----------------------------------------------------------------");
                CkPrintf("%s\n",out.str().c_str());
                CkExit();
            }
            else
                out<<"\n"<<std::setw(commNameLen)<<commName[curCommType];
        }
    }

    /// Delete the reduction message
    delete msg;
    /// If we're here, then simply trigger the next multicast
    sendMulticast(curCommType,curMsgSize);
}


#include "reductionBenchmark.def.h"


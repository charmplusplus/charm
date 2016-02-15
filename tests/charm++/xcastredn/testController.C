#include "testController.h"
#include <iomanip>

//----------------- externed globals -----------------
extern TestController *mainChare;
extern int bcastHandlerID;

/// Readonly proxy to the QHogger group
CProxy_QHogger hogger;
// Define the readonly config object
config cfg;
/// The names of the communication mechanisms being tested in this benchmark
char commName[][commNameLen] = {
                                 "CkMulticast-Bcast",
                                 "Charm-Bcast",
                                 "Converse-Bcast",
                                 "CkMulticast-Redn",
                                 "Charm-Redn",
                                 "Charm-SetRedn",
                                 "Converse-Redn",
                               };



/// An overloaded increment for comfort when handling the enum
inline CommMechanism operator++(CommMechanism &m)
{ return m = (CommMechanism)(m + 1); }



/// Function that fills the scheduler q and then triggers the test
void qFiller(void *param, void *msg)
{
    #ifdef VERBOSE_OPERATION
        CkPrintf("\n[%d] Filling scheduler Q with %d entry methods that have %d Mflop", CkMyPe(), cfg.qLength, cfg.flopM);
    #endif
    /// Build up a scheduler queue of the required length
    for (int i=0; i < cfg.qLength; i++)
        hogger[CkMyPe()].doSomething(cfg.flopM);
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
            cfg.qLength          = atoi(m->argv[4]);
        if (m->argc >= 6)
            cfg.flopM            = atoi(m->argv[5]);
    }
    else
        CkPrintf("Wrong number of arguments. Try %s numRepeats msgSizeMin(bytes) msgSizeMax(KB) qFillLength fillMethodDuration(us)",m->argv[0]);

    delete m;
    CkPrintf("\nMeasuring performance of chare array collectives using different communication libraries in charm++. \nNum PEs: %d \nTest parameters are: \n\tArray size = Section size = Num PEs = %d \n\tMsg sizes: %d bytes to %d KB \n\tNum repeats: %d \n\tScheduler Q Fill Length: %d entry methods \n\tScheduler Q Fill Method Total Flops: %d Mflop",
             CkNumPes(), cfg.arraySize, cfg.msgSizeMin, cfg.msgSizeMax, cfg.numRepeats, cfg.qLength, cfg.flopM);

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
    arraySections.push_back( createSection(cfg.useContiguousSection) );

    /// Delegate the section collectives to the communication libraries
    //                            CkMulticast
    arraySections[0].ckSectionDelegate(mgr);

    /// Setup the client at the root of the reductions
    CkCallback *cb = new CkCallback(CkIndex_TestController::receiveReduction(0),thisProxy);
    chareArray.ckSetReductionClient(cb);
    mgr->setReductionClient(arraySections[0],cb);

    /// Start off with the first comm type and the smallest message size
    curCommType    = bcastCkMulticast;
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

    /// Allow any required library mainchares (originally: comlib) to complete their initialization
    thisProxy.finishInit();
}




CProxySection_MyChareArray TestController::createSection(const bool isSectionContiguous)
{
    /// Determine the lower starting index of the section along each dimension
    int Xl = 0;
    /// Determine a step size based on whether a contiguous section is needed
    int dX = 1;
    if (!isSectionContiguous)
        dX = cfg.arraySize / cfg.sectionSize;
    /// Determine the upper index bounds for the section
    int Xu = (cfg.sectionSize - 1) * dX;
    CkAssert(cfg.arraySize >= Xu);
    /// Create the section
    return CProxySection_MyChareArray::ckNew(chareArray,Xl,Xu,dX);
}




void TestController::finishInit()
{
    /// Wait for quiescence and then start the timing tests
    CkCallback *trigger = new CkCallback(CkIndex_TestController::startTest(), thisProxy);
    CkCallback filler(qFiller, (void*)trigger);
    CkStartQD(filler);
}




void TestController::startTest()
{
    sendMulticast(curCommType, curMsgSize);
}




void TestController::sendMulticast(const CommMechanism commType, const int msgSize)
{
    /// Create a message of required size
    int numXcastUnits, numRednUnits;
    if (commType < rednCkMulticast)
    {
        numXcastUnits = curMsgSize/sizeof(double);
        numRednUnits  = 1;
    }
    else
    {
        numXcastUnits = 1;
        numRednUnits  = curMsgSize/sizeof(double);
    }

    #ifdef VERBOSE_STATUS
        CkPrintf("\nMsgSize: %f Sending out multicast number %d",(float)(numXcastUnits*sizeof(double))/1024,curRepeatNum+1);
    #endif

    DataMsg *msg = new (numXcastUnits) DataMsg(numXcastUnits, numRednUnits,commType);

    /// Fill it with data
    for (int i=0; i<numXcastUnits; i++)
        msg->data[i] = i;

    /// Start the timer and trigger the send to the array / section
    switch (commType)
    {
        case bcastCkMulticast:
            timeStart = CmiWallTimer();
            arraySections[0].crunchData(msg);
            break;

        case bcastCharm:
            timeStart = CmiWallTimer();
            chareArray.crunchData(msg);
            break;

        case bcastConverse:
        {
            DataMsg::pack(msg);
            envelope *env = UsrToEnv(msg);
            CmiSetHandler(env, bcastHandlerID);
            timeStart = CmiWallTimer();
            CmiSyncBroadcastAllAndFree(env->getTotalsize(), (char*)env);
            break;
        }

        case rednCkMulticast:
            timeStart = CmiWallTimer();
            arraySections[0].crunchData(msg);
            break;

        case rednCharm:
            timeStart = CmiWallTimer();
            arraySections[0].crunchData(msg);
            break;

        case setRednCharm:
            timeStart = CmiWallTimer();
            arraySections[0].crunchData(msg);
            break;

        case rednConverse:
            timeStart = CmiWallTimer();
            arraySections[0].crunchData(msg);
            break;

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
        CkPrintf("\nMsgSize: %f Received reduction number %d for repeat number %d", (float)msg->getSize()/1024, msg->getRedNo(), curRepeatNum+1);
    #endif

    /// If this is the first ever multicast/reduction loop, dont time it as it includes tree setup times etc
    if (curCommType == bcastCkMulticast && msg->getRedNo() == 0)
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
            if (++curCommType >= EndOfTest)
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


#include "reductionBenchmark.h"
#include <iomanip>
#include <math.h>

MyChareArray::MyChareArray(CkGroupID grpID): msgNum(0), mcastGrpID(grpID)
{
    #ifdef VERBOSE_CREATION
        CkPrintf("\n[%d,%d,%d] Just born...",thisIndex.x,thisIndex.y,thisIndex.z);
    #endif
    mcastMgr = CProxy_CkMulticastMgr(mcastGrpID).ckLocalBranch();
}



inline void MyChareArray::crunchData(DataMsg *msg)
{
    #ifdef VERBOSE_OPERATION
        CkPrintf("\n[%d,%d,%d] Received msg number %d",thisIndex.x,thisIndex.y,thisIndex.z,msgNum++);
    #endif
    /// Touch the data cursorily
    msg->data[0]++;
    /// Prepare some data to be returned
    double *returnData = new double(msg->size);
    /// Contribute to reduction
    #ifdef VERBOSE_OPERATION
        CkPrintf("\n[%d,%d,%d] Going to trigger reduction using mechanism: %s",thisIndex.x,thisIndex.y,thisIndex.z,commName[msg->commType]);
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

        default:
            CkAbort("Attempting to use unknown mechanism to handle the reduction");
            break;
    }
    delete msg;
}




Main::Main(CkArgMsg *m)
{
    //Process command-line arguments
    if (m->argc == 1)
    {
        // just proceed silently for empty args
    }
    else if ( ((m->argc>=2) && (m->argc<=5)) || (m->argc == 8) || (m->argc == 11) )
    {
        if (m->argc >= 2)
            cfg.numRepeats       = atoi(m->argv[1]);
        if (m->argc >= 3)
            cfg.msgSizeMin       = atoi(m->argv[2]);
        if (m->argc >= 4)
            cfg.msgSizeMax       = atoi(m->argv[3]);
        if (m->argc >= 5)
            cfg.useContiguousSection =(atoi(m->argv[4]) == 0)? false: true;
        if (m->argc >= 8)
        {
            cfg.section.X        = atoi(m->argv[5]);
            cfg.section.Y        = atoi(m->argv[6]);
            cfg.section.Z        = atoi(m->argv[7]);
        }
        if (m->argc == 11)
        {
            cfg.X                = atoi(m->argv[8]);
            cfg.Y                = atoi(m->argv[9]);
            cfg.Z                = atoi(m->argv[10]);
        }
    }
    else
        CkPrintf("Wrong number of arguments. Try %s numRepeats msgSizeMin msgSizeMax isSectionContiguous sectionDimX sectionDimY sectionDimZ arrayDimX arrayDimY arrayDimZ",m->argv[0]);

    delete m;
    CkPrintf("\nRunning timing tests for multicast/reductions using CkMulticast. Inputs are: \n\tArray size: (%d,%d,%d) \n\tSection size: (%d,%d,%d) \n\tMsg sizes (KB): %d to %d \n\tNum repeats: %d",
             cfg.X,cfg.Y,cfg.Z, cfg.section.X, cfg.section.Y, cfg.section.Z, cfg.msgSizeMin, cfg.msgSizeMax, cfg.numRepeats);

    // Setup the multicast manager stuff
    CkGroupID mcastGrpID  = CProxy_CkMulticastMgr::ckNew();
    CkMulticastMgr *mgr   = CProxy_CkMulticastMgr(mcastGrpID).ckLocalBranch();

    /// Create the array
    chareArray            = CProxy_MyChareArray::ckNew(mcastGrpID,cfg.X,cfg.Y,cfg.Z); 
    /// Create the array section
    createSection(cfg.useContiguousSection);

    /// Delegate the section collectives to the multicast manager
    arraySection.ckSectionDelegate(mgr);
    /// Setup the client at the roots of the reductions
    CkCallback *cb = new CkCallback(CkIndex_Main::receiveReduction(0),thisProxy);
    chareArray.ckSetReductionClient(cb);
    mgr->setReductionClient(arraySection,cb);

    /// Start off with the first comm type and the smallest message size
    curCommType    = CharmBcast;
    curMsgSize     = cfg.msgSizeMin;
    curRepeatNum   = 0;
    out<<std::fixed<<std::setprecision(6);
    out<<"\n"<<std::setw(cfg.fieldWidth)<<"Msg size (KB)"<<std::setw(cfg.fieldWidth)<<"Avg time (ms)";
    for (int i=1;i<=cfg.numRepeats;i++)
        out<<std::setw(cfg.fieldWidth-3)<<"Trial "<<std::setw(3)<<i;

    /// Send out the multicast
    sendMulticast(curCommType,curMsgSize);
}




void Main::createSection(const bool isSectionContiguous)
{
    /// Determing the lower starting index of the section along each dimension
    int Xl = 0, Yl = 0, Zl = 0;

    /// Determine a step size based on whether a contiguous section is needed
    int dX = 1, dY = 1, dZ = 1;
    if (!isSectionContiguous)
    {
        dX = floor( cfg.X/cfg.section.X );
        dY = floor( cfg.Y/cfg.section.Y );
        dZ = floor( cfg.Z/cfg.section.Z );
    }

    /// Determine the extent of the section along each dimension
    int Xu = (cfg.section.X-1)*dX;
    int Yu = (cfg.section.Y-1)*dY;
    int Zu = (cfg.section.Z-1)*dZ;
    CkAssert(cfg.X >= Xu && cfg.Y >= Yu && cfg.Z >= Zu);

    /// Create the section
    arraySection = CProxySection_MyChareArray::ckNew(chareArray,Xl,Xu,dX,Yl,Yu,dY,Zl,Zu,dZ);
}




void Main::sendMulticast(const CommMechanism commType, const int msgSize)
{
    #ifdef VERBOSE_STATUS
        CkPrintf("\nMsgSize: %d Sending out multicast number %d",curMsgSize,curRepeatNum+1);
    #endif
    /// Create a message of required size
    int numUnits = curMsgSize*1024/sizeof(double);
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
            arraySection.crunchData(msg);
            break;

        case Comlib:
            timeStart = CmiWallTimer();
            arraySection.crunchData(msg);
            break;

        default:
            CkAbort("Attempting to use unknown mechanism to communicate with chare array");
            break;
    }
}




void Main::receiveReduction(CkReductionMsg *msg)
{
    /// Compute the time taken (in milliseconds) for this multicast/reduction loop
    loopTimes.push_back( 1000*(CmiWallTimer() - timeStart) );

    #ifdef VERBOSE_STATUS
        CkPrintf("\nMsgSize: %d Received reduction number %d for repeat number %d",curMsgSize,msg->getRedNo(),curRepeatNum+1);
    #endif

    /// If this is the first ever multicast/reduction loop, dont time it as it includes tree setup times etc
    if (msg->getRedNo() == 0)
    {
        CkPrintf("\nFirst mcast/red loop took: %.6f ms. This includes tree setup times etc",loopTimes[0]);
        loopTimes.pop_back();
        curRepeatNum--;
    }

    /// If this ends the timings for a msg size
    if (++curRepeatNum >= cfg.numRepeats)
    {
        /// Compute the average time
        double avgTime = 0;
        for (int i=0; i< loopTimes.size(); i++)
            avgTime += loopTimes[i];
        avgTime /= loopTimes.size();
        /// Collate the results
        out<<"\n"<<std::setw(cfg.fieldWidth)<<curMsgSize<<std::setw(cfg.fieldWidth)<<avgTime;
        for (int i=0; i< loopTimes.size(); i++)
            out<<std::setw(cfg.fieldWidth)<<loopTimes[i];
        /// Reset counters, timings and sizes
        curRepeatNum = 0;
        curMsgSize *= 2;
        loopTimes.clear();

        /// If this ends the timings for all msg sizes, exit
        if (curMsgSize > cfg.msgSizeMax)
        {
            /// Print the results
            CkPrintf("\n----------------------------------------------------------------");
            CkPrintf("\nFinished timing the collectives mechanism: %s. Results: \n%s\n",commName[curCommType],out.str().c_str());
            /// Clear the output buffer
            out.str("");
            /// Reset the counters
            curMsgSize = cfg.msgSizeMin;
            /// Exit if done
            if (++curCommType >= Comlib)
                CkExit();
        }
    }

    /// Delete the reduction message
    delete msg;
    /// If we're here, then simply trigger the next multicast
    sendMulticast(curCommType,curMsgSize);
}


#include "reductionBenchmark.def.h"


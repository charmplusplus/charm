#include "myChareArray.h"
#include "convMsgHandlers.h"
#include <algorithm>

//----------------- externed globals -----------------
CpvExtern(std::vector<MyChareArray*>, localElems);



MyChareArray::MyChareArray(CkGroupID grpID): msgNum(0), mcastGrpID(grpID)
{
    #ifdef VERBOSE_CREATION
        CkPrintf("\nArrayElem[%d] Just born...",thisIndex);
    #endif
    mcastMgr = CProxy_CkMulticastMgr(mcastGrpID).ckLocalBranch();
    CpvAccess(localElems).push_back(this);
    /// Prepare some data to be returned (max sized contribution)
    int numUnits = cfg.msgSizeMax * 1024 /sizeof(double);
    returnData   = new double[numUnits];
    std::fill(returnData, returnData + numUnits, 100);
}




void MyChareArray::crunchData(DataMsg *msg)
{
    #ifdef VERBOSE_OPERATION
        CkPrintf("\nArrayElem[%d] Received msg number %d", thisIndex, msgNum++);
    #endif
    /// Touch the data cursorily
    msg->data[0] = 0;
    /// Contribute to reduction
    #ifdef VERBOSE_OPERATION
        CkPrintf("\nArrayElem[%d] Going to trigger reduction using mechanism: %s",thisIndex,commName[msg->commType]);
    #endif
    switch (msg->commType)
    {
        case bcastCkMulticast:
            CkGetSectionInfo(sid, msg);
            mcastMgr->contribute(msg->rednSize*sizeof(double),returnData,CkReduction::sum_double,sid);
            break;

        case bcastCharm:
            mcastMgr->contribute(msg->rednSize*sizeof(double),returnData,CkReduction::sum_double,sid);
            break;

        case bcastConverse:
            mcastMgr->contribute(msg->rednSize*sizeof(double),returnData,CkReduction::sum_double,sid);
            break;

        case rednCkMulticast:
            CkGetSectionInfo(sid, msg);
            mcastMgr->contribute(msg->rednSize*sizeof(double),returnData,CkReduction::sum_double,sid);
            break;

        case rednCharm:
            contribute(msg->rednSize*sizeof(double),returnData,CkReduction::sum_double);
            break;

        case setRednCharm:
            contribute(msg->rednSize*sizeof(double), returnData, CkReduction::set);
            break;

        case rednConverse:
        {
            CkReductionMsg *redMsg = CkReductionMsg::buildNew( msg->rednSize*sizeof(double), returnData, CkReduction::sum_double);
            CkReductionMsg::pack( redMsg );
            envelope *redEnv = UsrToEnv(redMsg);
            /// Contribute to reduction
            CmiSetHandler(redEnv, rednHandlerID);
            CmiReduce(redEnv, redEnv->getTotalsize(), convRedn_sum);
            break;
        }

        default:
            CkAbort("Attempting to use unknown mechanism to handle the reduction");
            break;
    }
    delete msg;
}


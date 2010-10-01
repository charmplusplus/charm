#include "myChareArray.h"

//----------------- externed globals -----------------
extern std::vector<MyChareArray*> localElems;



MyChareArray::MyChareArray(CkGroupID grpID): msgNum(0), mcastGrpID(grpID)
{
    #ifdef VERBOSE_CREATION
        CkPrintf("\nArrayElem[%d] Just born...",thisIndex);
    #endif
    mcastMgr = CProxy_CkMulticastMgr(mcastGrpID).ckLocalBranch();
    localElems.push_back(this);
}




void MyChareArray::crunchData(DataMsg *msg)
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


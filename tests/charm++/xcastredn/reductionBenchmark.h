#include "reductionBenchmark.decl.h"
#include "ckmulticast.h"

#include <vector>
#include <iostream>
#include <sstream>

// Debug macros
//#define VERBOSE_CREATION
//#define VERBOSE_OPERATION
//#define VERBOSE_STATUS

#ifndef REDUCTION_BENCHMARK_H
#define REDUCTION_BENCHMARK_H

/// Enumerate the different mechanisms for collective comm 
enum CommMechanism
{ CkMulticast, CharmBcast, ConverseBcast, ConverseToArrayBcast, Comlib};
/// The names of the communication mechanisms being tested in this benchmark
const int commNameLen = 30;
char commName[][commNameLen] = {"CkMulticast", "Charm-Bcast/Redn", "Converse-Bcast/Redn", "ConverseBcast/ArrayRedn", "Comlib"};




/// Utility structure that has the program settings
class config
{
    public:
        config(): fieldWidth(15) { setDefaults(); }

        void pup(PUP::er &p)
        {
            p|arraySize;  p|sectionSize;
            p|msgSizeMin; p|msgSizeMax;
            p|qLength;    p|uSecs;
            p|numRepeats;
            p|useContiguousSection;
        }

        void setDefaults()
        {
            arraySize   = CkNumPes();
            sectionSize = CkNumPes();
            msgSizeMin  = 8;
            msgSizeMax  = 64;
            qLength     = 0;
            uSecs       = 0;
            numRepeats  = 3;
            useContiguousSection = true;
        }

        /// Array and Section sizes
        int arraySize, sectionSize;
        /// Number of times to repeat the multicast/reduction for a single message size
        int numRepeats;
        /// The minimum msg (payload) size (in bytes)
        int msgSizeMin;
        /// The maximum msg (payload) size (in KB)
        int msgSizeMax;
        /// Is the section constructed out of randomly chosen array elements
        bool useContiguousSection;
        /// How long the scheduler q should be (num enqueued entry methods)
        int qLength;
        /// How long each entry method should be (micro seconds)
        int uSecs;
        /// Some output beauty
        const int fieldWidth;
} cfg; ///< readonly




/// A charm msg that is used in the xcast loop
class DataMsg: public CkMcastBaseMsg, public CMessage_DataMsg
{
    public:
        DataMsg(int numUnits, CommMechanism cType): size(numUnits), commType(cType) {}
        int size;
        CommMechanism commType;
        double *data;
};




/// A chare array that participates in the xcast/redn loop
class MyChareArray: public CBase_MyChareArray
{
    public: 
        MyChareArray(CkGroupID grpID);
        MyChareArray(CkMigrateMessage *msg) {}
        void pup(PUP::er &p) {}
        /// @entry Receives data and toys with it. Returns confirmation via a reduction
        void crunchData(DataMsg *msg);

    private:
        int msgNum;
        CkGroupID mcastGrpID;
        CkMulticastMgr *mcastMgr;
        CkSectionInfo sid;
};




/** A group that maintains its presence in the scheduler queue
 *
 * Used to mimic a real application scenario to measure the
 * performance degradation caused by having xcast msgs wait
 * in the scheduler q
 */
class QHogger: public CBase_QHogger
{
    public:
        QHogger(): numCalled(0) {}
        /// @entry An entry method that recursively hogs the scheduler Q
        void doSomething(int uSecs = 0)
        {
            numCalled++;
            /// Do something to hog some time
            usleep(uSecs);
            /// Renqueue myself to keep the scheduler busy
            thisProxy[CkMyPe()].doSomething(uSecs);
        }
    private:
        long int numCalled;
};




/// Test controller. Triggers and measures each xcast/redn loop
class Main: public CBase_Main
{
    public:
        Main(CkArgMsg *m);
        /// @entry Starts the timing tests
        void startTest() { sendMulticast(curCommType, curMsgSize); }
        /// @entry Reduction client method. Receives the result of the reduction
        void receiveReduction(CkReductionMsg *msg);

    private:
        /// Create an array section that I will multicast to
        CProxySection_MyChareArray createSection(const bool isSectionContiguous);
        /// Sends out a multicast to the array section
        void sendMulticast(const CommMechanism commType, const int msgSize);

        /// Chare array that is going to receive the multicasts
        CProxy_MyChareArray chareArray;
        /// Array section proxy
        std::vector<CProxySection_MyChareArray> arraySections;
        /// Counter for tracking the comm mechanism that is currently being tested
        CommMechanism curCommType;
        /// Counters for tracking test progress
        int curMsgSize,curRepeatNum;
        /// Stream holding all the results
        std::stringstream out, log;
        /// Start time for the multicast
        double timeStart;
        /// A vector (of size numRepeats) of times taken for a multicast/reduction loop
        std::vector<double> loopTimes;
};

/// An overloaded increment for comfort when handling the enum
inline CommMechanism operator++(CommMechanism &m)
{ return m = (CommMechanism)(m + 1); }

#endif // REDUCTION_BENCHMARK_H


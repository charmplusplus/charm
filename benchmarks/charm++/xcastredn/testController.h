#include "reductionBenchmark.h"

#ifndef TEST_CONTROLLER_H
#define TEST_CONTROLLER_H

/// Test controller. Triggers and measures each xcast/redn loop
class TestController: public CBase_TestController
{
    public:
        TestController(CkArgMsg *m);
        /// @entry Split the setup into two phases (originally, to accommodate libraries like comlib)
        void finishInit();
        /// @entry Starts the timing tests
        void startTest();
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

#endif // TEST_CONTROLLER_H


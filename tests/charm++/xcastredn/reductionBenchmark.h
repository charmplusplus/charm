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
{ CharmBcast, CkMulticast, Comlib };
/// The names of the collective mechanisms
char commName[][50] = {"Charm-Bcast/Redn","CkMulticast","Comlib"};



/// Utility structure that has the program settings
class config
{
    public:
        config(): 
                  numRepeats(3),
                  msgSizeMin(4), msgSizeMax(64),
                  useContiguousSection(false),
                  X(5), Y(5), Z(5),
                  fieldWidth(15)
        {
            section.X= 3; section.Y = 3; section.Z = 3;
        }

        void pup(PUP::er &p)
        {
            p|section.X; p|section.Y; p|section.Z;
            p|numRepeats;
            p|msgSizeMin;   p|msgSizeMax;
            p|useContiguousSection; 
            p|X; p|Y; p|Z;
        }

        /// Array section dimensions
        struct dims { int X, Y, Z; } section;
        /// Number of times to repeat the multicast/reduction for a single message size
        int numRepeats;
        /// The range of message sizes for which the benchmark should be run (in KB)
        int msgSizeMin, msgSizeMax;
        /// Is the section constructed out of randomly chosen array elements
        bool useContiguousSection;
        /// The dimensions of the chare array
        int X, Y, Z;
        /// Some output beauty
        const int fieldWidth;
} cfg; ///< readonly




class DataMsg: public CkMcastBaseMsg, public CMessage_DataMsg
{
    public:
        DataMsg(int numUnits, CommMechanism cType): size(numUnits), commType(cType) {}
        int size;
        CommMechanism commType;
        double *data;
};




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




class Main: public CBase_Main
{
    public:
        Main(CkArgMsg *m);
        /// @entry Reduction client method. Receives the result of the reduction
        void receiveReduction(CkReductionMsg *msg);

    private:
        /// Create an array section that I will multicast to
        void createSection(const bool isSectionContiguous);
        /// Sends out a multicast to the array section
        void sendMulticast(const CommMechanism commType, const int msgSize);

        /// Chare array that is going to receive the multicasts
        CProxy_MyChareArray chareArray;
        /// Array section proxy
        CProxySection_MyChareArray arraySection;
        /// Counter for tracking the comm mechanism that is currently being tested
        CommMechanism curCommType;
        /// Counters for tracking test progress
        int curMsgSize,curRepeatNum;
        /// Stream holding all the results
        std::stringstream out;
        /// Start time for the multicast
        double timeStart;
        /// A vector (of size numRepeats) of times taken for a multicast/reduction loop
        std::vector<double> loopTimes;
};

/// An overloaded increment for comfort when handling the enum
inline CommMechanism operator++(CommMechanism &m)
{ return m = (CommMechanism)(m + 1); }

#endif // REDUCTION_BENCHMARK_H


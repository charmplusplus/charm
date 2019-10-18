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

/// The different communication mechanisms being benchmarked
enum CommMechanism
{
    bcastCkMulticast,
    bcastCharm,
    bcastConverse,
    rednCkMulticast,
    rednCharm,
    setRednCharm,
    rednConverse,
    EndOfTest
};

const int commNameLen        = 30;
/// The names of all the comm mechanisms (used in output)
extern char commName[][commNameLen];



/// Utility structure that has the program settings
class config
{
    public:
        config(): fieldWidth(15) { setDefaults(); }

        void pup(PUP::er &p)
        {
            p|arraySize;  p|sectionSize;
            p|msgSizeMin; p|msgSizeMax;
            p|qLength;    p|flopM;
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
            flopM       = 0;
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
        int flopM;
        /// Some output beauty
        const int fieldWidth;
};

/// readonly
extern config cfg;




/// A charm msg that is used in the xcast loop
class DataMsg: public CkMcastBaseMsg, public CMessage_DataMsg
{
    public:
        DataMsg(int numXcastUnits, int numRednUnits, CommMechanism cType)
               : size(numXcastUnits), rednSize(numRednUnits), commType(cType) {}
        int size, rednSize;
        CommMechanism commType;
        double *data;
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
        void doSomething(int flopM = 0)
        {
            numCalled++;
            /// Do something to hog some time
            double a = 1;
            for (int i=0; i < 1000000 * flopM; i++)
                a = 0.5*a + 0.5;
            /// Renqueue myself to keep the scheduler busy
            thisProxy[CkMyPe()].doSomething(flopM);
        }
    private:
        long int numCalled;
};

#endif // REDUCTION_BENCHMARK_H


#include "client.decl.h"
#include "mylib.h"
#define CK_TEMPLATES_ONLY
#include "mylib.def.h"
#undef CK_TEMPLATES_ONLY

#include "utils.h"
#include <sstream>
#include <functional>
#include <iterator>

typedef count< std::less<libdtype> > cntType;
// Temporary initproc to register the instantiated EPs
void register_instantiations()
{
    CkReductionTarget(pgm, acceptResults<avg>);
    CkReductionTarget(pgm, acceptResults< cntType >);
};


// reducer type definitions
CkReduction::reducerType countReducer;
CkReduction::reducerType avgReducer;

// Register reducer functions
void register_reducers()
{
    countReducer = CkReduction::addReducer(cntType::reduce_count);
    avgReducer   = CkReduction::addReducer(avg::reduce_avg);
}


// Test driver
class pgm : public CBase_pgm
{
    public:
        pgm(CkArgMsg *m): nElements(2 * CkNumPes()), nDatumsPerChare(1000), nDone(0)
        {
            CkPrintf("[main] Creating a library chare array with %d chares and %d datums per chare\n",
                    nElements, nDatumsPerChare);
            // Create the library chare array
            arrProxy = CProxy_libArray::ckNew(nDatumsPerChare, nElements);
            thisProxy.startTest();
            delete m;
        }
        
        void startTest() {
            // Run the tests
            // Setup a redn cb and start the parallel sum computation of sum and avg
            CkCallback avgCB(CkReductionTarget(pgm, acceptResults<avg>), thisProxy);
            arrProxy.doSomething(avg(), avgReducer, avgCB);

            // Setup a redn cb and start the parallel count of num elements less than given threshold
            CkCallback cntCB(CkReductionTarget(pgm, acceptResults< cntType >), thisProxy);
            arrProxy.doSomething( cntType(0.5), countReducer, cntCB );
        }

        template <typename T>
        void acceptResults(T op) {
            std::ostringstream out;
            out << "[main] Applied operation to all data in library chare array in parallel:\n"
                << op
                << "\n";
            CkPrintf("%s", out.str().c_str());
            if (++nDone == 2)
                CkExit();
        }

    private:
        CProxy_libArray arrProxy;
        int nElements, nDone, nDatumsPerChare;
};

#define CK_TEMPLATES_ONLY
#include "client.def.h"
#undef CK_TEMPLATES_ONLY
#include "client.def.h"


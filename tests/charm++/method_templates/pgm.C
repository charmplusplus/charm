#include "client.decl.h"
#include "mylib.h"
#define CK_TEMPLATES_ONLY
#include "mylib.def.h"
#undef CK_TEMPLATES_ONLY

#include "utils.h"
#include <sstream>
#include <functional>
#include <iterator>

// Temporary initproc to register the instantiated EPs
void register_instantiations()
{
    count< std::less<int> >  comparator;
    avg avger;
    CkReduction::reducerType foo;
    CkCallback bar;
    CkIndex_libArray::doSomething(comparator, foo, bar);
    CkIndex_libArray::doSomething(avger, foo, bar);
};


CkReduction::reducerType countReducer;
CkReduction::reducerType avgReducer;

// Register reducer functions
void register_reducers()
{
    countReducer = CkReduction::addReducer(count< std::less<int> >::reduce_count);
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
            //arrProxy.doSomething( count< std::less<int> >(5) );

            // Setup a redn cb and start the parallel sum computation
            CkCallback avgCB(CkReductionTarget(pgm, avgDone), thisProxy);
            arrProxy.doSomething(avg(), avgReducer, avgCB);
        }

        void avgDone(avg avger) {
            std::ostringstream out;
            out << "[main] Summed all data in the library chare array:\n"
                << avger
                << "\n";
            CkPrintf("%s", out.str().c_str());
            endTest();
        }

        void endTest() {
            if (++nDone == 1)
                CkExit();
        }

    private:
        CProxy_libArray arrProxy;
        int nElements, nDone, nDatumsPerChare;
};

#include "client.def.h"


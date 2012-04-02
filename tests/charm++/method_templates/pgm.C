#include "client.decl.h"
#include "mylib.h"
#define CK_TEMPLATES_ONLY
#include "mylib.def.h"
#undef CK_TEMPLATES_ONLY

#include "utils.h"
#include <iostream>
#include <functional>
#include <iterator>

// Temporary initproc to register the instantiated EPs
void register_instantiations()
{
    count< std::less<int> >  comparator;
    avg avger;
    CkIndex_libArray::doSomething(comparator);
    CkIndex_libArray::doSomething(avger);
};


// Register reducer functions
void register_reducers()
{
    CkReduction::reducerType countReducer = CkReduction::addReducer(count< std::less<int> >::reduce_count);
    CkReduction::reducerType avgReducer   = CkReduction::addReducer(avg::reduce_avg);
}


// Test driver
class pgm : public CBase_pgm
{
    public:
        pgm(CkArgMsg *m): nElements(2 * CkNumPes()), nDone(0)
        {
            // Create the library chare array and configure a reduction client
            arrProxy = CProxy_libArray::ckNew(1000, nElements);
            arrProxy.ckSetReductionClient( new CkCallback(CkIndex_pgm::endTest(), thisProxy) );
            thisProxy.startTest();
            delete m;
        }
        
        void startTest() {
            // Run the tests
            arrProxy.doSomething( count< std::less<int> >(5) );
            arrProxy.doSomething(avg());
        }

        void endTest() {
            if (++nDone == 1)
                CkExit();
        }

    private:
        CProxy_libArray arrProxy;
        int nElements, nDone;
};

#include "client.def.h"


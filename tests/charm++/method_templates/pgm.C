#include "client.decl.h"
#include "mylib.h"
#define CK_TEMPLATES_ONLY
#include "mylib.def.h"
#undef CK_TEMPLATES_ONLY
#include <iostream>

// Utility functors
class add {
    private:
        int sum;
    public:
        add(): sum(0) {}
        inline void operator() (int i) { sum += i; }
        void pup(PUP::er &p) { p | sum; }
        friend std::ostream& operator<< (std::ostream& out, const add& obj) {
            out << "sum = " << obj.sum;
            return out;
        }
};


class avg {
    private:
        int sum, num;
    public:
        avg(): sum(0), num(0) {}
        inline void operator() (int i) { sum += i; num++; }
        void pup(PUP::er &p) { p | sum; p | num; }
        friend std::ostream& operator<< (std::ostream& out, const avg& obj) {
            out << "num = " << obj.num << "; "
                << "avg = " << ( obj.num ? obj.sum/obj.num : obj.sum );
            return out;
        }
};


// Temporary initproc to register the instantiated EPs
void register_instantiations()
{
    add adder;
    avg avger;
    CkIndex_libArray::doSomething<add>(adder);
    CkIndex_libArray::doSomething<avg>(avger);
};


// Test driver
class pgm : public CBase_pgm
{
    public:
        pgm(CkArgMsg *m): nElements(CkNumPes()), nDone(0)
        {
            arrProxy = CProxy_libArray::ckNew(nElements);
            arrProxy.ckSetReductionClient( new CkCallback(CkIndex_pgm::endTest(), thisProxy) );
            thisProxy.startTest();
            delete m;
        }
        
        void startTest() {
            arrProxy.doSomething(add());
            arrProxy.doSomething(avg());
        }

        void endTest() {
            if (++nDone == 2)
                CkExit();
        }

    private:
        CProxy_libArray arrProxy;
        int nElements, nDone;
};

#include "client.def.h"


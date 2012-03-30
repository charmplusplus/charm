#include "client.decl.h"
#include "mylib.h"
#define CK_TEMPLATES_ONLY
#include "mylib.def.h"
#undef CK_TEMPLATES_ONLY
#include <iostream>
#include <functional>

// Utility functors
template <typename cmp>
class count {
    private:
        int threshold, num;
        cmp c;
    public:
        count(const int _t=0): threshold(_t), num(0) {}
        inline void operator() (int i) { if (c(i, threshold)) num++; }
        void pup(PUP::er &p) { p | threshold; p | num; }
        friend std::ostream& operator<< (std::ostream& out, const count& obj) {
            out << "threshold = "<< obj.threshold << "; "
                << "num = " << obj.num;
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
                << "sum = " << obj.sum << "; "
                << "avg = " << ( obj.num ? (double)obj.sum/obj.num : obj.sum );
            return out;
        }
};


// Temporary initproc to register the instantiated EPs
void register_instantiations()
{
    count< std::less<int> >  comparator;
    avg avger;
    CkIndex_libArray::doSomething< count<std::less<int> > >(comparator);
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
            //count< std::less<int> > cnt(5);
            arrProxy.doSomething( count< std::less<int> >(5) );
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


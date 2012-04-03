#ifndef UTILS_H
#define UTILS_H

#include "charm++.h"
#include <iostream>

typedef double libdtype;

// A pup-friendly functor identical to return value of std::bind2nd(cmp(), threshold)
template <typename cmp>
class count {
    private:
        libdtype threshold;
        int num;
        cmp c;
    public:
        //
        count(const libdtype _t=0): threshold(_t), num(0) {}

        // Operate on an input element
        inline void operator() (libdtype* first, libdtype *last)
        {
            for (libdtype *ptr = first; ptr != last; ptr++)
                if (c(*ptr, threshold)) num++;
        }

        // Serialize the internals
        void pup(PUP::er &p) { p | threshold; p | num; }

        // Spit results to ostream
        friend std::ostream& operator<< (std::ostream& out, const count& obj) {
            out << "threshold = "<< obj.threshold << "; "
                << "num = " << obj.num;
            return out;
        }

        // Reducer function to accumulate across multiple count objects
        static CkReductionMsg* reduce_count(int nMsg, CkReductionMsg **msgs)
        {
            CkAssert(nMsg > 0);
            count<cmp>* result = (count<cmp>*) msgs[0]->getData();
            for (int i = 1; i < nMsg; ++i)
            {
                count<cmp> *contrib = (count<cmp>*) msgs[i]->getData();
                // Compare the thresholds themselves and use the appropriate one
                if ( result->c(contrib->threshold, result->threshold) )
                    result->threshold = contrib->threshold;
                result->num += contrib->num;
            }
            return CkReductionMsg::buildNew(sizeof(count<cmp>), result);
        }
};


// Functor that computes the sum and avg of a sequence of integers
class avg {
    private:
        libdtype sum;
        int num;
    public:
        avg(): sum(0), num(0) {}

        // Operate on an input element
        inline void operator() (libdtype* first, libdtype *last)
        {
            num += std::distance(first, last);
            for (libdtype *ptr = first; ptr != last; ptr++)
                sum += *ptr;
        }

        // Serialize the internals
        void pup(PUP::er &p) { p | sum; p | num; }

        // Spit results to ostream
        friend std::ostream& operator<< (std::ostream& out, const avg& obj) {
            out << "num = " << obj.num << "; "
                << "sum = " << obj.sum << "; "
                << "avg = " << ( obj.num ? (double)obj.sum/obj.num : obj.sum );
            return out;
        }

        // Reducer function to accumulate across multiple avg objects
        static CkReductionMsg* reduce_avg(int nMsg, CkReductionMsg **msgs)
        {
            CkAssert(nMsg > 0);
            avg* result = (avg*) msgs[0]->getData();
            for (int i = 1; i < nMsg; ++i)
            {
                avg *contrib = (avg*) msgs[i]->getData();
                result->sum += contrib->sum;
                result->num += contrib->num;
            }
            return CkReductionMsg::buildNew(sizeof(avg), result);
        }
};

#endif // UTILS_H

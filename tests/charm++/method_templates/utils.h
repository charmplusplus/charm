#ifndef UTILS_H
#define UTILS_H

// A pup-friendly functor identical to return value of std::bind2nd(cmp(), threshold)
template <typename cmp>
class count {
    private:
        int threshold, num;
        cmp c;
    public:
        //
        count(const int _t=0): threshold(_t), num(0) {}
        // Operate on an input element
        inline void operator() (int* first, int *last)
        {
            for (int *ptr = first; ptr != last; ptr++)
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
};


// Functor that computes the sum and avg of a sequence of integers
class avg {
    private:
        int sum, num;
    public:
        avg(): sum(0), num(0) {}
        // Operate on an input element
        inline void operator() (int* first, int *last)
        {
            num += std::distance(first, last);
            for (int *ptr = first; ptr != last; ptr++)
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
};



#endif // UTILS_H

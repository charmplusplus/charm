#include<iostream>
#include<stdio.h>
#include<assert.h>

using namespace std;

class Double {
    int getNumElements() {
        int i=0;
        Double *iter = this;
        while(iter!=0) {
            i++;
            iter = iter->next;
        }
        return i;
    }

public:
    double data;
    Double *next;

    Double()
    {
        data = 0.0;
        next = 0;
    }

    // copy constructor
    //
	// Differs from copy assignment because cc deals with
	// unallocated memory, but ca deals with a constructed object.
    Double(const Double &rhs)
    {
        data = rhs.data;
        next = 0;

        // call assignment operator
        *this = rhs;
    }

    ~Double()
    {
//         cout << "reached destructor" << endl;
        delete next;
    }

    // assignment operator
    Double& operator= (const Double& rhs)
    {
        if (this == &rhs) return *this;  // self-assignment

        if (next != 0) {
            delete next;
            next = 0;
        }

        Double *iter1 = this;
        const Double *iter2 = &rhs;
        while (iter2 != 0) {
            iter1->data = iter2->data;
            if (iter2->next != 0)
                iter1->next = new Double();
            iter2 = iter2->next;
            iter1 = iter1->next;
        }

        return *this;
    }

    // += operator
    // @@ what if rhs is a sequence.  Do we want to prepend the entire sequence to this Double?
    Double& operator+= (const Double& rhs)
    {
        Double *tmp = new Double(data);
        tmp->next = next;
        data = rhs.data;
        next = tmp;

        return *this;
    }

    // typecast from int
    Double(const int rhs) : data(rhs), next(0)
    {
//         cout << "reached typecast from int" << endl;
    }

    // pup
//     virtual void pup(PUP::er &p){
//         int n = getNumElements();
//         p|n;

//         Double *iter = this;
//         if (p.isUnpacking()) {
//             while (n>0) {
//                 p|(iter->data);
//                 n--;
//                 if (n>0)
//                     iter->next = new Double();
//                 iter = iter->next;
//             }
//         } else {
//             while(iter!=0) {
//                 p|(iter->data);
//                 iter = iter->next;
//             }
//         }
//     }

    // typecast Double from/to double, for convenience
    Double(const double &rhs) : data(rhs), next(0) {}
//     operator double() { return data; }
//     operator double const () { return (const double) data; }
};

// convenience function
ostream& operator << (ostream& os, const Double& s) {
    os << s.data;
    if (s.next!=0)
        os << " " << *(s.next);
    return os;
}

int
main()
{
    Double d;  // default constructor
    cout << d << endl;
    Double e(1); // typecast from int
    d += e; // +=
    cout << d << endl;
    d += d; // prepends only the head
    cout << "d=" << d << endl;
    Double *f = new Double(d); // =
    cout << "f=" << *f << endl;  // destructor
    delete f;
    for(int i=3; i<200; i++)
        d += i;
    cout << d << endl;
    cout << "the end" << endl;
}

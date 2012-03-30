#ifndef MYLIB_H
#define MYLIB_H

#include "mylib.decl.h"
#include <sstream>

extern int moduleRo;

class libArray: public CBase_libArray
{
    public:
        libArray() {}
        libArray(CkMigrateMessage *msg) {}

        // Allows client to visit and operate on library-owned data
        // @note: Clients should supply type that provides void operator() (int) and an ostream inserter
        template <typename T>
        void doSomething(T t)
        {
            // Apply client specified operation to my chunk of data
            for (int i = thisIndex*10; i< (thisIndex+1)*10; i++)
                t(i);
            // Do something with the result
            std::ostringstream out;
            out << "\nlibArray[" << thisIndex << "] " << t;
            CkPrintf("%s", out.str().c_str());
            // Notify completion
            contribute();
        }
};

#endif // MYLIB_H


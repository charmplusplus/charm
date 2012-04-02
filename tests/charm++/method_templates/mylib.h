#ifndef MYLIB_H
#define MYLIB_H

#include "mylib.decl.h"
#include <sstream>

extern int moduleRo;

class libArray: public CBase_libArray
{
    public:
        libArray(int _arrSize): arrSize(_arrSize), myData(NULL)
        {
            myData = new int[arrSize];
            for (int i = 0; i < arrSize; i++)
                myData[i] = thisIndex * arrSize + i;
        }

        libArray(CkMigrateMessage *msg) {}

        // @note: Clients should supply type T that provides:
        // void operator() (int *begin, int*end)
        // ostream& operator<< (ostream& out, const T& obj)
        /// Allows client to visit and operate on library-owned data
        template <typename T>
        void doSomething(T t)
        {
            // Apply client specified operation to my chunk of data
            t(myData, myData + arrSize);
            // Do something with the result
            std::ostringstream out;
            out << "\nlibArray[" << thisIndex << "] " << t;
            CkPrintf("%s", out.str().c_str());
            // Notify completion
            contribute();
        }

    private:
        int arrSize;
        int *myData;
};

#endif // MYLIB_H


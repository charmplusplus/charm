#ifndef MYLIB_H
#define MYLIB_H

#include "mylib.decl.h"

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
        void doSomething(T t, CkReduction::reducerType redType, CkCallback redCB)
        {
            // Apply client specified operation to my chunk of data
            t(myData, myData + arrSize);
            // Contribute any intelligence the client has gathered into a reduction
            contribute(sizeof(t),&t,redType,redCB);
        }

    private:
        int arrSize;
        int *myData;
};

#endif // MYLIB_H


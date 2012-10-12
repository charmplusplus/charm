#include "sync_square.decl.h"

class int_message : public CMessage_int_message {
    public:
        int value;
        int_message(int val) : value(val) {}
};

class Driver : public CBase_Driver {
    public: 
        Driver(CkArgMsg*);
        void get_square(int value);
        CProxy_Squarer s;
};

class Squarer : public CBase_Squarer {
    public:
        Squarer(void) {}
        Squarer(CkMigrateMessage* m) {}
        int_message* square(int x);
};


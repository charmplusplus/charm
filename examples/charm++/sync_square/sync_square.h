#include "sync_square.decl.h"

class int_message : public CMessage_int_message {
    public:
        int value;
        int_message(int val) : value(val) {}
};

class Driver : public CBase_Driver {
    public: 
        Driver(CkArgMsg*);
        void get_square(int val);
        CProxy_Squarer s;
        CProxy_SquarerArr sarr;
        CProxy_SquarerGrp sgrp;
};

class Squarer : public CBase_Squarer {
    public:
        Squarer(void) {}
        Squarer(CkMigrateMessage* m) {}
        int square(int x);
        int_message* squareM(int x);
};


class SquarerArr : public CBase_SquarerArr {
    public:
        SquarerArr(void) {}
        SquarerArr(CkMigrateMessage* m) {}
        int square(int x);
        int_message* squareM(int x);
};


class SquarerGrp : public CBase_SquarerGrp {
    public:
        SquarerGrp(void) {}
        SquarerGrp(CkMigrateMessage* m) {}
        int square(int x);
        int_message* squareM(int x);
};

#include "MainOutputVcSelection.h"

class outputBuffer : public OutputVcSelection {
        public:
        int selectOutputVc(map<int,int> & availBuffer,const Packet *h,const int);
};


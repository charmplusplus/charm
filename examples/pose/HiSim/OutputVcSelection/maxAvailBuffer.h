#include "MainOutputVcSelection.h"

class maxAvailBuffer : public OutputVcSelection {
        public:
        int selectOutputVc(map<int,int> & availBuffer,const Packet *h,int);
};


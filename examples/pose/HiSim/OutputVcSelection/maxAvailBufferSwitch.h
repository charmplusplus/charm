#include "MainOutputVcSelection.h"

class maxAvailBufferSwitch : public OutputVcSelection {
        public:
        int selectOutputVc(map<int,int> & availBuffer,map<int,int> & mapVc,const Packet *h);
};


#include "MainOutputVcSelection.h"

class maxAvailBufferBubbleVc : public OutputVcSelection {
        public:
        int selectOutputVc(map<int,int> & availBuffer,const Packet *h);
};


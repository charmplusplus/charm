#include "outputBuffer.h"

// Return vcid within a port ( not like selectInputVc which returns within a switch )
int outputBuffer::selectOutputVc(map<int,int> & Bufsize,const Packet *h,const int inputVc) {
		CkAssert(inputVc < config.switchVc);
                return inputVc;
}

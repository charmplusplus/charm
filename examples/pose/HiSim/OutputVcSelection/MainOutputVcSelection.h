
#ifndef __OUTPUTVCSELECTION_H
#define __OUTPUTVCSELECTION_H
#include "../Main/BgSim_sim.h"
class OutputVcSelection {
        public:
        virtual int selectOutputVc(map<int,int> & Bufsize,const Packet *h){}
};
#endif

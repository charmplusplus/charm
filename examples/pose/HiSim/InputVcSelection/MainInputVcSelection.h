
#ifndef __INPUTVCSELECTION_H
#define __INPUTVCSELECTION_H
#include "../Main/BgSim_sim.h"
class InputVcSelection {
        public:
        virtual int selectInputVc(map<int,int> & availBuffer,map<int,int> & request,map<int,vector <Header> > &inBuffer,const int)=0;
};
#endif

#include "MainInputVcSelection.h"

class SLQ_Switch : public InputVcSelection {
        public:
        int selectInputVc(map<int,int> & Bufsize,map<int,int> & requested,map<int, vector <Header> > & inBuffer,const int);
};

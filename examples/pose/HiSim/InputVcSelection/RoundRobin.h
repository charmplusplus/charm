#include "MainInputVcSelection.h"

class RoundRobin : public InputVcSelection {
        public:
	int roundRobin;
	RoundRobin() { roundRobin = 0; }
        int selectInputVc(map<int,int> & Bufsize,map<int,int> & requested,map<int, vector <Header> > & inBuffer, const int ignore );
};


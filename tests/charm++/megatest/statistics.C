/*
 Statistics and random number test program
  Orion Sky Lawlor, olawlor@acm.org, 11/26/2002
*/
#include "statistics.h"
#include "ckstatistics.h"

void checkRange(double v,double lo,double hi,const char *desc) {
	if (v<lo || v>hi) {
		CkError("Error: expected %s %g to be between %g and %g!\n",
			desc,v,lo,hi);
		CkAbort("Statistic out of range!");
	}
}

void statistics_init(void) {
	//Try out the Converse random number generator
	CkSample s;
	for (int i=0;i<10000;i++) s+=CrnDrand();
	checkRange(s.getMean(),0.4,0.6,"mean"); //Theory: 0.5
	checkRange(s.getStddev(),0.2,0.4,"stddev"); //Theory: 0.2887=sqrt(1/12)
	checkRange(s.getMin(),0.0,0.1,"min"); //Theory: 0
	checkRange(s.getMax(),0.9,1.0,"max"); //Theory: 1
	
	megatest_finish();
}
void statistics_moduleinit(void) {}

// Need a fake "_register" routine because we don't have a .ci file...
void _registerstatistics(void) {}
MEGATEST_REGISTER_TEST(statistics,"olawlor",1)

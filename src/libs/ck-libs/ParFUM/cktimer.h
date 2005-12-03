#ifndef CMK_THRESHOLD_TIMER
#define CMK_THRESHOLD_TIMER

/** Time a sequence of operations, printing out the
names and times of any operations that exceed a threshold. 

Use it with only the constructor and destructor like:
      void foo(void) {
          CkThresholdTimer t("foo");
	  ...
      }
(this times the whole execution of the routine,
all the way to t's destructor is called on function return)

Or, you can start different sections like:
      void bar(void) {
          CkThresholdTimer t("first");
	  ...
	  t.start("second");
	  ...
	  t.start("third");
	  ...
      }
  
This class *only* prints out the time if it exceeds
a threshold-- by default, one millisecond.
*/
class CkThresholdTimer {
	double threshold; // Print any times that exceed this (s).
	double lastStart; // Last activity started at this time (s).
	const char *lastWhat; // Last activity has this name.
	
	void start_(const char *what) {
		lastStart=CmiWallTimer();
		lastWhat=what;
	}
	void done_(void) {
		double elapsed=CmiWallTimer()-lastStart;
		if (elapsed>threshold) {
			CmiPrintf("%s took %.2f s\n",lastWhat,elapsed);
		}
	}
public:
	CkThresholdTimer(const char *what,double thresh=0.001) 
		:threshold(thresh) { start_(what); }
	void start(const char *what) { done_(); start_(what); }
	~CkThresholdTimer() {done_();}
};

#endif

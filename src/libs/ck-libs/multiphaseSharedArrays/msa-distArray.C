// emacs mode line -*- mode: c++; tab-width: 4 -*-
#include "charm++.h"
#include "msa-distArray.h"

MSA_Listener::~MSA_Listener() {}

MSA_Listeners::MSA_Listeners() {}
	
/// Add this listener to your set.
void MSA_Listeners::add(MSA_Listener *l) {
	l->add();
	listeners.push_back(l);
}

/// Signal all added listeners and remove them from the set.
void MSA_Listeners::signal(unsigned int pageNo) {
	if (listeners.size()>0) {
		for (unsigned int i=0;i<listeners.size();i++)
			listeners[i]->signal(pageNo);
		listeners.resize(0);
	}
}

MSA_Listeners::~MSA_Listeners() {
	if (listeners.size()!=0)
		CkAbort("Tried to delete MSA_Listeners before signaling!");
}

/// Wait for one more page.
void MSA_Thread_Listener::add(void) {
	count++;
}

/// If we're waiting for any pages, suspend our thread.
void MSA_Thread_Listener::suspend(void) {
	bool verbose=false;
	if (count>0) {
		thread=CthSelf();
		if (verbose) CkPrintf("Thread %p suspending for %d signals\n",
			CthSelf(),count);
		CthSuspend();
		if (verbose) CkPrintf("Thread %p resumed\n",CthSelf());
	}
}

void MSA_Thread_Listener::signal(unsigned int pageNo)
{
	count--;
	if (count<0) CkAbort("MSA_Thread_Listener signaled more times than added!");
	if (count==0) { /* now ready to resume thread (if needed) */
		if (thread!=0) {
			CthAwaken(thread);
			thread=0;
		}
	}
}

#include "msa.def.h"

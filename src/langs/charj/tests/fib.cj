package tests;
import charj.lang.*;

public class Fib extends Chare {
    Fib parent_;
	boolean root_;
	int n_;
	int partialResult_;
	int pendingChildren_;
	
	entry Fib(boolean root, int n, Fib parent) {
	  CkPrintf("in fib::fib. n=%d\n", n);
	  n_ = n;
	  root_ = root;
	  parent_ = parent;

	  if (n <= 2) {
	      partialResult_ = 1;
	      passUp();
	  }
	  partialResult_ = 0;
	  pendingChildren_ = 2;
	  Fib child1 = new Fib(false, n-1, thisHandle); // replace with "@this"?
	  Fib child2 = new Fib(false, n-2, thisHandle); // can we get away with just "this"?
	}
	
	entry void passUp(int subTreeValue) {
	    partialResult += subTreeValue;
	    if (pendingChildren_) return;
	    if (root_) {
	        CkPrintf("Fib(%d) = %d", n_, partialResult_);
	        CkExit();
	    }
	    parent.passUp(partialResult_); // replace with passUp(partialResult_)@parent;
	}
}

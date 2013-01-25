
package tests;

readonly Main@ main;
readonly int num_elements;

extern atoi;

public mainchare Main {
    public entry Main(CkArgMsg[~]@ m) {
        //Process command-line arguments
        num_elements = 5;
        if (m.argc > 1) num_elements = atoi(m.argv[1]);
        delete m;

        //Start the computation
        CkPrintf("Running Hello on %d processors for %d elements\n",
                 CkNumPes(), num_elements);
        main = thisProxy;
        Hello@ arr = new Hello@(num_elements);
        arr[0]@SayHi(17);
    }

    public entry void done() {
        CkPrintf("All done\n");
        CkExit();
    }
}

public chare_array [1D] Hello {
    public entry Hello() {
        CkPrintf("Hello %d created\n", thisIndex);
    }
    
    public entry void SayHi(int hi_num) {
        CkPrintf("Hi[%d] from element %d\n",hi_num,thisIndex);
        if (thisIndex < num_elements-1) {
            //Pass the hello on:
            thisProxy[thisIndex+1]@SayHi(hi_num+1);
        } else {
            //We've been around once-- we're done.
            main@done();
        }
    }
}

package tests.unit;

readonly Main@ main_proxy;
// TODO: allow this assignment
//readonly int num_rows = 5;
//readonly int num_cols = 5;
readonly int num_rows;
readonly int num_cols;

public mainchare Main {
    int num_checkins = 0;

    public entry Main(CkArgMsg m) {
        delete m;
        Cell@ cell_array;
        // TODO: this should not be needed.
        num_rows = 5;
        num_cols = 5;

        CkPrintf("Running Parallel on %d processors for %d elements\n",
            CkNumPes(),
            num_rows*num_cols);
        main_proxy = thishandle;
        cell_array = new Cell@(num_rows, num_cols);
        cell_array@finished(100);
        cell_array@process();
    }

    public entry void end() {
        if (++num_checkins == num_rows*num_cols) CkExit();
    }
}

public chare_array [2d] Cell {
    int val;

    public entry Cell() {
        //val = thisIndex.x*10 + thisIndex.y;
        val = 10;
    }

    public sdagentry void finished(int number);

    public sdagentry void process() {
        CkPrintf("Migrating from PE %d\n", CkMyPe());
        thisProxy@process2();
        this.migrateMe(0);
    }

    public entry void process2() {
        when finished [100] (int number) {
            CkPrintf("(%d, %d): received finished message\n",
                    1,1);//thisIndex.x,
                    //thisIndex.y);
            main_proxy@end();
        }
    }
}

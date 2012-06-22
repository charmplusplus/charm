
extern atoi;
extern CkAbort;

readonly Main@ main;
readonly Chunk@ chunks;
readonly int num_finished;

mainchare Main {
    public entry Main(CkArgMsg[~]@ m) {
        if (m.argc != 4) CkAbort("Wrong parameters\n");

        int x = atoi(m.argv[1]);
        int y = atoi(m.argv[2]);
        int k = atoi(m.argv[3]);

        if (x < k) CkAbort("Xdim must be greater than k");
        if (k < CkNumPes() || k % CkNumPes()) CkAbort("k must be a multiple of numPes.");
        
        chunks = new Chunk@(k, x, y, k);
        chunks@jacobi();
        num_finished = 0;
    }
}


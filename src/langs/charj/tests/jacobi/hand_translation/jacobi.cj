
extern atoi;
extern CkAbort;

readonly Main@ main_proxy;
readonly Chunk@ chunks;
readonly double start_time;

mainchare Main {
    int num_finished = 0;
    entry Main(CkArgMsg[~]@ m) {
        if (m.argc != 4) CkAbort("Wrong parameters\n");

        int x = atoi(m.argv[1]);
        int y = atoi(m.argv[2]);
        int k = atoi(m.argv[3]);

        if (x < k) CkAbort("Xdim must be greater than k");
        if (k < CkNumPes() || k % CkNumPes()) CkAbort("k must be a multiple of numPes.");
        
        chunks = new Chunk@(k, x, y, k);
        chunks@jacobi();
        num_finished = 0;
        start_time = CmiWallTimer();
        main_proxy = thisProxy;
    }

    entry void finished() {
        if (++num_finished == CkNumPes()) {
            double elapsed = CmiWallTimer() - start_time;
            CkPrintf("Finished in %fs %fs/step, %d iterations\n", elapsed, elapsed/_sdag_jacobi_i, _sdag_jacobi_i);
            CkPrintf("Numfin=%d, total=%d, Pes = %d\n", _sdag_jacobi_i, total, CkNumPes());
            CkExit();
        }
    }
}


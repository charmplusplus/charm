class Jacobi: public CBase_Jacobi {
  Jacobi_SDAG_CODE

  public:
    int iterations;
    int imsg;
#if  JACOBI
    double *new_temperature;
#endif
    double *temperature;
	double timing,average;
    int neighbors;
    int thisIndex_x, thisIndex_y, thisIndex_z;
    
    Jacobi() ;
    Jacobi(CkMigrateMessage* m): CBase_Jacobi(m) { }
    ~Jacobi() { 
        delete [] temperature; }

    void begin_iteration(void) ;
    void processGhosts(ghostMsg *gmsg) ;
	void check_and_compute() ;
    double compute_kernel() ;     //Gauss-Siedal compute
    void constrainBC() ;
    void print();
	void ResumeFromSync();
    void pup(PUP::er &p);
};


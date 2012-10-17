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
    // Constructor, initialize values
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
	// Pupping function for migration and fault tolerance
	// Condition: assuming the 3D Chare Arrays are NOT used
	
	void ResumeFromSync();
    void pup(PUP::er &p);
    //void doStep();
};


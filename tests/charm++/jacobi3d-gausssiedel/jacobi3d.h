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

class JacobiNodeMap : public CkArrayMapObj {
PUPable_decl(JacobiNodeMap);
private:
    int X, Y, Z;
public:
    JacobiNodeMap(int x, int y, int z) : X(x), Y(y), Z(z) {}
    JacobiNodeMap(CkMigrateMessage* m) {}
    ~JacobiNodeMap() {}
    void pup(PUP::er& p) {
      p | X;
      p | Y;
      p | Z;
    }

    int homePe(const CkArrayIndex &idx) const {
      int *index = (int *)idx.data();
      return (CkMyNodeSize() * (index[0]*X*Y + index[1]*Y + index[2]))%CkNumPes();
    }
};

// TODO: Especially under the new mapping framework, these maps should just get
// the array size data from the array options in the setArrayOptions call.
class JacobiMap : public CkArrayMapObj {
PUPable_decl(JacobiMap);
  public:
    int X, Y, Z;
    int *mapping;

    JacobiMap(int x, int y, int z);
    JacobiMap(CkMigrateMessage* m) {}
    ~JacobiMap() {
        delete [] mapping;
    }
    void pup(PUP::er& p) {
      p | X;
      p | Y;
      p | Z;
      if (p.isUnpacking()) mapping = new int[X*Y*Z];
      PUParray(p, mapping, X*Y*Z);
    }

    int homePe(const CkArrayIndex &idx) const {
        int *index = (int *)idx.data();
        return mapping[index[0]*Y*Z + index[1]*Z + index[2]];
    }
};

#define MAXDIMENSION 3
class bitvec
{
 public:
  unsigned short vec[MAXDIMENSION];
  short  numbits;
  bitvec() {
    vec[0] = 0;
    vec[1] = 0;
    vec[2] = 0;
    numbits = 0;
  }
  bitvec(unsigned short *in_vec , short nobits){
    memset(vec, 0,(sizeof(unsigned short)*MAXDIMENSION));
    memcpy(vec, in_vec, (sizeof (unsigned short)*MAXDIMENSION));
    numbits = nobits ;
  }
  void pup(PUP::er &p){
    p(vec, MAXDIMENSION);
    p(numbits);
  }
};

			 
class CkArrayIndexBitVec : public CkArrayIndex 
{
 public:
  bitvec vecIndex;
  CkArrayIndexBitVec(){}
  CkArrayIndexBitVec(const bitvec &in) {
    vecIndex = in;
    nInts = sizeof(vecIndex)/sizeof(int);
  }
};

#define MAXDIMENSION 3
class BitVec
{
 public:
  unsigned short vec[MAXDIMENSION];
  short  numbits;
  BitVec() {
    vec[0] = 0;
    vec[1] = 0;
    vec[2] = 0;
    numbits = 0;
  }
  BitVec(unsigned short *in_vec , short nobits){
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
  BitVec vecIndex;
  CkArrayIndexBitVec(){}
  CkArrayIndexBitVec(const BitVec &in) {
    vecIndex = in;
    nInts = sizeof(vecIndex)/sizeof(int);
  }
};

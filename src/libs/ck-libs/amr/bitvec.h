#include <array>

constexpr int MAXDIMENSION = 3;

class BitVec
{
public:
  std::array<unsigned short, MAXDIMENSION> vec;
  short numbits;
  BitVec()
  {
    vec.fill(0);
    numbits = 0;
  }
  BitVec(const BitVec& in)
  {
    vec = in.vec;
    numbits = in.numbits;
  }
  void pup(PUP::er& p)
  {
    p | vec;
    p | numbits;
  }
};

class CkArrayIndexBitVec : public CkArrayIndex
{
private:
  BitVec* vecIndex;

public:
  CkArrayIndexBitVec(const BitVec& in)
  {
    vecIndex = new (index) BitVec(in);
    nInts = sizeof(BitVec) / sizeof(int);
  }
};

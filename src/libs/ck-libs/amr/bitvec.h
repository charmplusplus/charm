#define MAXDIMENSION 3
#if CK_ARRAYINDEX_MAXLEN < MAXDIMENSION+1
#  warning "You need to set CK_ARRAYINDEX_MAXLEN to at least 4"
#endif

class bitvec
{
	public:
		unsigned int vec[MAXDIMENSION];
		int  numbits;
		bitvec() {
		  vec[0] = 0;
		  vec[1] = 0;
		  vec[2] = 0;
		  numbits = 0;
		}
		bitvec(unsigned short *in_vec , int nobits)
		{
		  memset(vec, 0,(sizeof(unsigned short)*MAXDIMENSION));
		  memcpy(vec, in_vec, (sizeof (unsigned short)*MAXDIMENSION));
		  //vec[2] = 0;
		  numbits = nobits ;
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
		/*	virtual const unsigned char *getKey(int &nbytes) const 
		{
			nbytes = sizeof(bitvec);
			return (const unsigned char *) &vecIndex;
			}*/
};

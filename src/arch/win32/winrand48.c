#define RAND48_SEED_0        0x330e
#define RAND48_SEED_1        0xabcd
#define RAND48_SEED_2        0x1234
#define RAND48_ADD           0xb
#define RAND48_MULT_0        0xe66d
#define RAND48_MULT_1	     0xdeec
#define RAND48_MULT_2        0x0005
#define N                    16

/* RAND48_MULT_2 might be 0x000f*/ 

static unsigned short __rand48_seed[3] = { RAND48_SEED_0, RAND48_SEED_1, RAND48_SEED_2 };
static unsigned short __rand48_mult[3] = { RAND48_MULT_0, RAND48_MULT_1, RAND48_MULT_2 };
static unsigned short __rand48_add = RAND48_ADD;

extern "C"
void srand48(long seed)
{
        __rand48_seed[0] = RAND48_SEED_0;
        __rand48_seed[1] = (unsigned short) seed;
        __rand48_seed[2] = (unsigned short) (seed >> 16);
        __rand48_mult[0] = RAND48_MULT_0;
        __rand48_mult[1] = RAND48_MULT_1;
        __rand48_mult[2] = RAND48_MULT_2;
        __rand48_add = RAND48_ADD;
}


extern "C"
static void __dorand48(unsigned short xseed[3])
{
        unsigned long accu;
        unsigned short temp[2];

        accu = (unsigned long) __rand48_mult[0] * (unsigned long) xseed[0] +
         (unsigned long) __rand48_add;
        temp[0] = (unsigned short) accu;        /* lower 16 bits */
        accu >>= sizeof(unsigned short) * 8;
        accu += (unsigned long) __rand48_mult[0] * (unsigned long) xseed[1] +
         (unsigned long) __rand48_mult[1] * (unsigned long) xseed[0];
        temp[1] = (unsigned short) accu;        /* middle 16 bits */
        accu >>= sizeof(unsigned short) * 8;
        accu += __rand48_mult[0] * xseed[2] + __rand48_mult[1] * xseed[1] + __rand48_mult[2] * xseed[0];
        xseed[0] = temp[0];
        xseed[1] = temp[1];
        xseed[2] = (unsigned short) accu;
}


extern "C"
long lrand48(void)
{
    __dorand48(__rand48_seed);
    return ((long) __rand48_seed[2] << 15) + ((long) __rand48_seed[1] >> 1);
}

extern "C"
double drand48(void)
{
	double two16m = 1.0/(1L << N);
	__dorand48(__rand48_seed);

	return (two16m * (two16m * (two16m * __rand48_seed[0] + __rand48_seed[1]) + __rand48_seed[2]));

}
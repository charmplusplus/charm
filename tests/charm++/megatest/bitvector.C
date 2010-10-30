#define DEBUGGING
#include <charm++.h>
#include "bitvector.h"
#include "ckbitvector.h"
#include "megatest.h"

/* This class definition is garbage, to fill the .ci file so it doesn't
 * complain.
 */
class bitvectorMessage : public CMessage_bitvectorMessage {
};

bool test(bool result, const char *what) {
  if ( result ) {
//    ckout << what << " passed." << endl;
  } else {
    ckerr << what << " failed." << endl;
    megatest_finish();
  }

  return result;
}

void testBitFlipping() {
  int i;
  CkBitVector bvOrig(sizeof(unsigned int)*8),
            bv = bvOrig;

  // Test setting all the bits of bv based on the bits in value
  for(i=0;i<sizeof(unsigned int)*8;i++) {
    if ( i >= 16 ) {
      bv.Set(i);
    } else {
      bv.Clear(i);
    }
  }
  test((bv.getData())[0]==0xffff0000, "Set and Clear (low bits)");

  // Do that test again but on high bits.
  bv = CkBitVector(sizeof(unsigned int)*8*2);
  for(i=0;i<sizeof(unsigned int)*8;i++) {
    if ( i>=16 ) {
      bv.Set(i+sizeof(unsigned int)*8);
    } else {
      bv.Clear(i+sizeof(unsigned int)*8);
    }
  }
  test((bv.getData())[1]==0, "Set and Clear (low bits)");
  test((bv.getData())[0]==0xffff0000, "Set and Clear (high bits)");

  // Comparisons of the first chunk
  bv.ShiftDown(sizeof(unsigned int)*8);
  test((bv.getData())[1]==0xffff0000, "Shift Down (low bits)");
  test((bv.getData())[0]==0, "Shift Down (high bits)");

  // Comparisons of higher chunks
  bv.ShiftUp(sizeof(unsigned int)*8/2);
  test((bv.getData())[1]==0, "Shift Up (low bits)");
  test((bv.getData())[0]==0x0000ffff, "Shift Up (high bits)");

  // Set all bits true
  bv = bvOrig; bv.Invert();
  test((bv.getData())[0]==(~(0x0)), "One's compliment, all 0s");

  // Negation
  bv.Invert();
  test((bv.getData())[0]==(0x0), "One's compliment, mix");
}

void testUnion() {
  unsigned int a = 0x8a82, b=0x8511;
  CkBitVector bv1(0x8a82, 0x10000),
              bv2(0x8511, 0x10000);

  // Test the intersection
  bv1.Union(bv2);
  a = a | b;
  a = a << (sizeof(unsigned int)*8/2);
  test((bv1.getData())[0]==a, "Union in a chunk");
}

void testInter() {
  unsigned int a = 0x8a82, b=0x8511;
  CkBitVector bv1(0x8a82, 0x10000),
              bv2(0x8511, 0x10000);

  // Test the intersection
  bv1.Intersection(bv2);
  a = a & b;
  a = a << (sizeof(unsigned int)*8/2);
  test((bv1.getData())[0]==a, "Intersection in a chunk");
}

void testDiff() {
  unsigned int a = 0xffff, b=0x8511;
  CkBitVector bv1(0xffff, 0x10000),
              bv2(0x8511, 0x10000);

  // Test the difference
  bv1.Difference(bv2);
  a = a & ~b;
  a = a << (sizeof(unsigned int)*8/2);
  test((bv1.getData())[0]==a, "Difference in a chunk");
}

void testConcat() {
  unsigned int a = 0xff0000ff, b=0x00ffff00, c=0xff00ff00;
  CkBitVector bv1(0xff00,0x10000),
              bv2(0x00ff,0x10000),
	      t;

  // Test the two ways of concatting them.
  t = bv1; t.Concat(bv2);
  test((t.getData())[0]==a, "Concat under a chunk");

  // Now go over a chunk size in concatenation
  t.Concat(bv2); t.Concat(bv1);
  test((t.getData())[0]==a, "Concat over a chunk (low)");
  test((t.getData())[1]==b, "Concat over a chunk (high)");

  // Now try a chunk to itself.
  t = bv1; t.Concat(t);
  test((t.getData())[0]==c, "Concat to yourself");

  // Try a bv to itself larger than a single chunk
  t.Concat(t);
  test((t.getData())[0]==c, "Concat to yourself (low)");
  test((t.getData())[1]==c, "Concat to yourself (high)");
}

void testOps() {
  testBitFlipping();
  testUnion();
  testInter();
  testDiff();
  testConcat();
}

/* ************************************************************************
 *
 * ************************************************************************ */
void bitvector_init(void) {
  testOps();
  megatest_finish();
}

void bitvector_moduleinit(void){}

MEGATEST_REGISTER_TEST(bitvector,"jbooth",0)
#include "bitvector.def.h"

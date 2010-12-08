#include "Array.h"
#include <iostream>
#include <list>
#include <string>
#include <limits>

using namespace std;
using namespace CharjArray;

class ArrayTest {
public:
  virtual string testName() = 0;
  virtual bool test() = 0;
};

class Array1DUnit : public ArrayTest {
public:
  string testName() {
    return string("1D int array test");
  }

  bool test() {
    int size = 10;
    Range r(size);
    Array<int> *array1 = new Array<int>(Domain<1>(Range(size)));
    Array<int> &ref = *array1;

    if (ref.size() != size)
      return false;

    for (int i = 0; i < ref.size(); i++) {
      ref[i] = 20 + i;
    }

    for (int i = 0; i < size; i++) {
      if (ref[i] != i + 20)
        return false;
      //cout << i << ": " << ref[i] << endl;
    }

    return true;
  }
};

class Array2DUnitLinear : public ArrayTest {
public:
  string testName() {
    return string("2D int array test");
  }

  bool test() {
    int size1 = 2;
    int size2 = 3;
    Range dim1(size1);
    Range dim2(size2);
    Range rarray [2] = {dim1, dim2};
    Array<int, 2> *array1 = new Array<int, 2>(Domain<2>(rarray));
    Array<int, 2> &ref = *array1;

    if (ref.size() != size1 * size2)
      return false;

    for (int i = 0; i < ref.size(); i++) {
      ref[i] = 20 + i;
    }

    for (int i = 0; i < size1 * size2; i++) {
      if (ref[i] != i + 20)
        return false;
      //cout << i << ": " << ref[i] << endl;
    }

    return true;
  }
};

class Array2DUnitRowMajor : public ArrayTest {
public:
  string testName() {
    return string("2D int array test");
  }

  bool test() {
    int size1 = 2;
    int size2 = 3;
    Range dim1(size1);
    Range dim2(size2);
    Range rarray [2] = {dim1, dim2};
    Array<int, 2> *array1 = new Array<int, 2>(Domain<2>(rarray));
    Array<int, 2> &ref = *array1;

    if (ref.size() != size1 * size2)
      return false;

    for (int i = 0; i < ref.size(0); i++) {
      for (int j = 0; j < ref.size(1); j++) {
        ref.access(i, j) = i;
      }
    }

    for (int i = 0; i < size2; i++) {
      if (ref[i] != 0)
        return false;
    }

    for (int i = size2; i < size2 * 2; i++) {
      if (ref[i] != 1)
        return false;
    }

    return true;
  }
};

struct Slicing : public ArrayTest {
  string testName() {
    return "array slicing";
  }

  bool test() {
    const static int N = 5;
    Array<int> a = Domain<1>(N);
    for(int i = 0; i < N; ++i)
      a[i] = i;
    Array<int> &b = *(a[Domain<1>(3,4)]);
    if (b[0] != 3) {
      return false;
    }
    return true;
  }
};

template <typename T>
struct VectorTest : public ArrayTest {
  string testName() {
    return "vector math test";
  }

  bool test() {
    const static int N = 5;
    Vector<T> ones(N), zeroes(N), inc(N);
    for (int i = 0; i < N; ++i) {
      zeroes[i] = 0;
      ones[i] = 1;
      inc[i] = i+1;
    }

    if (dot(&ones,&ones) != ones.size())
      return false;

    if (dot(&ones,&zeroes) != 0)
      return false;

    if (dot(&inc, &ones) != N*(N+1)/2)
      return false;

    if (norm1(&zeroes) != 0 || norm1(&ones) != N || norm1(&inc) != N*(N+1)/2)
      return false;

    if (!norm2test(zeroes, ones))
      return false;

    if (normI(&zeroes) != 0 || normI(&ones) != 1 || normI(&inc) != N)
      return false;

    if (ones == zeroes || ones == inc || zeroes == inc)
      return false;
    if (ones != ones || zeroes != zeroes || inc != inc)
      return false;

    return true;
  }

  bool norm2test(const Vector<T> &zeroes, const Vector<T> &ones) {
    T norm = norm2(&ones);
    T deltaY = norm*norm - dot(&ones, &ones);
    // XXX: Dirty hack, no good for ints
    if (norm2(&zeroes) != 0 || deltaY > 10*numeric_limits<T>::epsilon()) {
      return false;
    }
    return true;
  }
};

template <>
string VectorTest<float>::testName() {
  return "CBLAS float test";
}

template <>
string VectorTest<double>::testName() {
  return "CBLAS double test";
}

template <typename T, class atype>
struct MatrixTest : public ArrayTest {
  string testName() {
    return "Matrix test";
  }

  bool test() {
    const static int N = 5;
    Matrix<T, atype> zeroes(N), ones(N);
    const Matrix<T, atype> &ident = *(Matrix<T, atype>::ident(N));
    Vector<T> v0(N), v1(N);

    v0.fill(0);
    v1.fill(1);
    zeroes.fill(0);
    ones.fill(1);

    for (int i = 0; i < N; ++i) {
      if(v1[i] != 1 || v0[i] != 0)
	return false;

      for (int j = 0; j < N; ++j) {
	if(ones.access(i,j) != 1 || zeroes.access(i,j) != 0)
	  return false;

	if ((ident.access(i,j) == 1) != (i == j))
	  return false;
      }
    }

    if (zeroes == ones || zeroes == ident || ones == ident)
      return false;
    if (ones != ones || zeroes != zeroes || ident != ident)
      return false;

    return true;
  }
};

static void print_result(ArrayTest *test)
{
  bool status = test->test();
  cout << test->testName() << (status ? " PASSED" : " FAILED") << endl;
}

int main(void) {
  list<ArrayTest*> tests;
  Array1DUnit a1;
  Array2DUnitLinear a2;
  Array2DUnitRowMajor a3;
  Slicing s;
  VectorTest<int> v;
  VectorTest<float> f;
  VectorTest<double> d;
  MatrixTest<double, RowMajor<2> > m;

  tests.push_back(&a1);
  tests.push_back(&a2);
  tests.push_back(&a3);
  tests.push_back(&s);
  tests.push_back(&v);
  tests.push_back(&f);
  tests.push_back(&d);
  tests.push_back(&m);

  for_each(tests.begin(), tests.end(), print_result);

#if 0
  Array<int>* newTest = new Array<int>(Domain<1>(Range(10)));

  for (int i = 0; i < 9; i++) {
    (*newTest)[i] = i;
  }

  for (int i = 0; i < 10; i++) {
    cout << "newTest: " << (*newTest)[i] << endl;
  }

  Array<int>* newTest2 = (*newTest)[Domain<1>(Range(5, 9))];

  for (int i = 0; i < newTest2->size(); i++) {
    cout << "newTest2: " << (*newTest2)[i] << endl;
  }
#endif

  return 0;
}

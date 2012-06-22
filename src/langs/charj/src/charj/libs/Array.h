#ifndef ARRAY_H
#define ARRAY_H

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <charm++.h>
#if CMK_HAS_CBLAS
#include <cblas.h>
#endif

namespace CharjArray {
  class Range {
  public:
    int size, start, stop;
    Range() {}
    Range(int size_) : size(size_), start(0), stop(size) {}
    Range(int start_, int stop_) :
    size(stop_ - start_), start(start_), stop(stop_) {
      assert(stop >= start);
    }
    void pup(PUP::er& p) { 
        p | size;
        p | start;
        p | stop;
    }
  };

  template<int dims>
  class Domain {
  public:
    Range ranges[dims];
    
    Domain() {}

    Domain(Range ranges_[]) {
      for (int i = 0; i < dims; i++) 
	ranges[i] = ranges_[i];      
    }

    Domain(Range range) {
      ranges[0] = range;
    }

    Domain(Range range1, Range range2) {
      // TODO: fix Charj generator so it uses the array
      ranges[0] = range1;
      ranges[1] = range2;
    }

    int size() const {
      int total = 0;

      for (int i = 0; i < dims; i++)
	if (total == 0)
	  total = ranges[i].size;
	else
	  total *= ranges[i].size;

      return total;
    }

    void pup(PUP::er& p) { 
        for (int i=0; i<dims; ++i) p | ranges[i];
    }
  };

  template<int dims>
  class RowMajor {
  public:
    static int access(const int i, const Domain<dims> &d) {
      return i - d.ranges[0].start;
    }
    static int access(const int i, const int j, const Domain<dims> &d) {
      return (i - d.ranges[0].start) * d.ranges[1].size + j -
        d.ranges[1].start;
    }
    // Generic access method, not used right now.
    // static int access(const int *i, const Domain<dims> &d) {
    //   int off = i[0];
    //   int dimoff = 1;
    //   for (int j = ndims-1; j > 0; --j) {
    //     dimoff *= d.ranges[j].size;
    //     off += dimoff * (i[j] - d.ranges[j].start);
    //   }
    //   return off;
    // }
  };

  template<int dims>
  class ColMajor {
  public:
    static int access(const int i, const Domain<dims> &d) {
      return i - d.ranges[0].start;
    }
    static int access(const int i, const int j, const Domain<dims> &d) {
      return (j - d.ranges[1].start) * d.ranges[1].size + i -
        d.ranges[0].start;
    }
  };

  template<class type, int dims = 1, class atype = RowMajor<dims> >
  class Array {
  private:
    Domain<dims> domain;
    type *block;
    int ref_cout;
    bool did_init;
    Array* ref_parent;

  public:
    Array(Domain<dims> domain_) : ref_parent(0), did_init(false) {
      init(domain_);
    }

    Array(type **block_) : did_init(false) {
      block = *block_;
    }

    Array(type& block_) : did_init(false) {
      block = &block_;
    }

    Array() : ref_parent(0), did_init(false) {

    }

    Array(Array* parent, Domain<dims> domain_)
        : ref_parent(parent), did_init(false) {
      domain = domain_;
      block = parent->block;
    }

    void init(Domain<dims> &domain_) {
      domain = domain_;
      //if (atype == ROW_MAJOR)
      block = new type[domain.size()];
      //printf("Array: allocating memory, size=%d, base pointer=%p\n",
      //       domain.size(), block);
      did_init = true;
    }

    type* raw() { return block; }

    ~Array() {
      if (did_init) delete[] block;
    }

    /*type* operator[] (const Domain<dims> &domain) {
      return block[domain.ranges[0].size];
      }*/

    type& operator[] (const int i) {
      return block[atype::access(i, domain)];
    }

    const type& operator[] (const int i) const {
      return block[atype::access(i, domain)];
    }

    type& access(const int i) {
      return this->operator[](i);
    }

    type& access(const int i, const int j) {
      //printf("Array: accessing, index (%d,%d), offset=%d, base pointer=%p\n",
      //i, j, atype::access(i, j, domain), block);
      return block[atype::access(i, j, domain)];
    }

    type& access(const int i, const Range r) {
      Domain<1> d(r);
      //printf("Array: accessing subrange, size = %d, range (%d,%d), base pointer=%p\n",
      //d.size(), r.start, r.stop, block);
      type* buf = new type[d.size()];
      for (int j = 0; j < d.size(); j++) {
        //printf("Array: copying element (%d,%d), base pointer=%p\n", i, j, block);
        buf[j] = block[atype::access(i, j, domain)];
      }
      return *buf;
    }

    const type& access(const int i, const int j) const {
      return block[atype::access(i, j, domain)];
    }

    Array<type, dims, atype>* operator[] (const Domain<dims> &domain) {
      return new Array<type, dims, atype>(this, domain);
    }

    int size() const {
      return domain.size();
    }

    int size(int dim) const {
      return domain.ranges[dim].size;
    }

    void pup(PUP::er& p) { 
        p | domain;
        if (p.isUnpacking()) {
            block = new type[domain.size()];
        }
        PUParray(p, block, domain.size());
    }

    void fill(const type &t) {
      for (int i = 0; i < domain.size(); ++i)
	block[i] = t;
    }

    /// Do these arrays have the same shape and contents?
    bool operator==(const Array &rhs) const
    {
      for (int i = 0; i < dims; ++i)
	if (this->size(i) != rhs.size(i))
	  return false;

      for (int i = 0; i < this->size(); ++i)
	if (this->block[i] != rhs.block[i])
	  return false;

      return true;
    }
    bool operator!=(const Array &rhs) const
    {
      return !(*this == rhs);
    }
  };

  /**
     A local Matrix class for various sorts of linear-algebra work.

     Indexed from 0, to reflect the C-heritage of Charj.
   */
  template <typename V, class atype = RowMajor<2> >
  class Matrix : public Array<V, 2, atype>
  {
  public:
    Matrix() { }
    /// A square matrix
    Matrix(unsigned int n) : Array<V,2,atype>(Domain<2>(n,n)) { }

    /// A identity matrix
    static Matrix* ident(int n)
    {
      Matrix *ret = new Matrix(n);
      ret->fill(0);

      for (int i = 0; i < n; ++i)
	ret->access(i,i) = 1;

      return ret;
    }
  };

  template <typename T, class atype = RowMajor<1> >
  class Vector : public Array<T, 1, atype>
  {
  public:
    Vector() { }
    Vector(unsigned int n) : Array<T, 1, atype>(Range(n)) { }
  };

  /// Compute the inner (dot) product v1^T * v2
  // To compute v1^H * v2, call as dot(v1.C(), v2)
  template<typename T, class atype1, class atype2>
  T dot(const Vector<T, atype1> *pv1, const Vector<T, atype2> *pv2)
  {
    const Vector<T, atype1> &v1 = *pv1, &v2 = *pv2;
    assert(v1.size() == v2.size());
    // XXX: This default initialization worries me some, since it
    // won't necessarily be an additive identity for all T. - Phil
    T ret = T();
    int size = v1.size();
    for (int i = 0; i < size; ++i)
      ret += v1[i] * v2[i];
    return ret;
  }
#if CMK_HAS_CBLAS
  template <>
  float dot<float, RowMajor<1>, RowMajor<1> >(const Vector<float, RowMajor<1> > *pv1,
					      const Vector<float, RowMajor<1> > *pv2)
  {
    const Vector<float, RowMajor<1> > &v1 = *pv1, &v2 = *pv2;
    assert(v1.size() == v2.size());
    return cblas_sdot(v1.size(), &(v1[0]), 1, &(v2[0]), 1);
  }
  template <>
  double dot<double, RowMajor<1>, RowMajor<1> >(const Vector<double, RowMajor<1> > *pv1,
						const Vector<double, RowMajor<1> > *pv2)
  {
    const Vector<double, RowMajor<1> > &v1 = *pv1, &v2 = *pv2;
    assert(v1.size() == v2.size());
    return cblas_ddot(v1.size(), &(v1[0]), 1, &(v2[0]), 1);
  }
#endif

  /// Computer the 1-norm of the given vector
  template<typename T, class atype>
    T norm1(const Vector<T, atype> *pv)
  {
    const Vector<T, atype> &v = *pv;
    // XXX: See comment about additive identity in dot(), above
    T ret = T();
    int size = v.size();
    for (int i = 0; i < size; ++i)
      ret += v[i];
    return ret;
  }

  /// Compute the Euclidean (2) norm of the given vector
  template<typename T, class atype>
    T norm2(const Vector<T, atype> *pv)
  {
    const Vector<T, atype> &v = *pv;
    // XXX: See comment about additive identity in dot(), above
    T ret = T();
    int size = v.size();
    for (int i = 0; i < size; ++i)
      ret += v[i] * v[i];
    return sqrt(ret);
  }
#if CMK_HAS_CBLAS
  template<>
    float norm2<float, RowMajor<1> >(const Vector<float, RowMajor<1> > *pv)
  {
    const Vector<float, RowMajor<1> > &v = *pv;
    return cblas_snrm2(v.size(), &(v[0]), 1);
  }
  template<>
    double norm2<double, RowMajor<1> >(const Vector<double, RowMajor<1> > *pv)
  {
    const Vector<double, RowMajor<1> > &v = *pv;
    return cblas_dnrm2(v.size(), &(v[0]), 1);
  }
#endif

  /// Compute the infinity (max) norm of the given vector
  // Will fail on zero-length vectors
  template<typename T, class atype>
    T normI(const Vector<T, atype> *pv)
  {
    const Vector<T, atype> &v = *pv;
    T ret = v[0];
    int size = v.size();
    for (int i = 1; i < size; ++i)
      ret = max(ret, v[i]);
    return ret;
  }

  /// Scale a vector by some constant
  template<typename T, typename U, class atype>
    void scale(const T &t, Vector<U, atype> *pv)
  {
    const Vector<T, atype> &v = *pv;
    int size = v.size();
    for (int i = 0; i < size; ++i)
      v[i] = t * v[i];
  }
#if CMK_HAS_CBLAS
  template<>
    void scale<float, float, RowMajor<1> >(const float &t,
					   Vector<float, RowMajor<1> > *pv)
  {
    Vector<float, RowMajor<1> > &v = *pv;
    cblas_sscal(v.size(), t, &(v[0]), 1);
  }
  template<>
    void scale<double, double, RowMajor<1> >(const double &t,
					     Vector<double, RowMajor<1> > *pv)
  {
    Vector<double, RowMajor<1> > &v = *pv;
    cblas_dscal(v.size(), t, &(v[0]), 1);
  }
#endif

  /// Add one vector to a scaled version of another
  template<typename T, typename U, class atype>
    void axpy(const T &a, const Vector<U, atype> *px, Vector<U, atype> *py)
  {
    Vector<T, atype> &x = *px;
    const Vector<T, atype> &y = *py;
    int size = x.size();
    assert(size == y.size());
    for (int i = 0; i < size; ++i)
      x[i] = a * x[i] + y[i];
  }
#if CMK_HAS_CBLAS
  template<>
    void axpy<float, float, RowMajor<1> >(const float &a,
					  const Vector<float, RowMajor<1> > *px,
					  Vector<float, RowMajor<1> > *py)
  {
    const Vector<float, RowMajor<1> > &x = *px;
    Vector<float, RowMajor<1> > &y = *py;
    int size = x.size();
    assert(size == y.size());
    cblas_saxpy(size, a, &(x[0]), 1, &(y[0]), 1);
  }
  template<>
    void axpy<double, double, RowMajor<1> >(const double &a,
					  const Vector<double, RowMajor<1> > *px,
					  Vector<double, RowMajor<1> > *py)
  {
    const Vector<double, RowMajor<1> > &x = *px;
    Vector<double, RowMajor<1> > &y = *py;
    int size = x.size();
    assert(size == y.size());
    cblas_daxpy(size, a, &(x[0]), 1, &(y[0]), 1);
  }
#endif
}

#endif

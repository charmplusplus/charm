#ifndef ARRAY_H
#define ARRAY_H

namespace CharjArray {
  class Range {
  public:
    int size, offset;
    Range() {}
    Range(int size_) : size(size_) {}
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

    int size() {
      int total = 0;
      for (int i = 0; i < dims; i++)
	if (total == 0)
	  total = ranges[i].size;
	else
	  total *= ranges[i].size;
    }
  };

  template<int dims>
  class Point : public Domain<dims> {
  public:
    Point(int dim) : Domain<dims>() {
      this->ranges[0].size = dim;
    }
  };

  enum ArrayType { RECT, JAGGED, ROW_MAJOR, COL_MAJOR };
 
  template<class type, int dims = 1, ArrayType atype = RECT>
  class Array {
  private:
    Domain<dims> domain;
    type *block;

  public:
    Array(Domain<dims> domain_) {
      init(domain_);
    }

    void init(Domain<dims> &domain_) {
      domain = domain_;
      if (atype == RECT)
	block = new type[domain.size()];
    }

    ~Array() {
      delete block;
    }
    
    type* operator[] (const Domain<dims> &domain) {
      return block[domain.ranges[0].size];
    }

    type& operator[] (const Point<dims> &point) {
      return block[point.ranges[0].size];
    }

    type& operator[] (const int index) {
      return block[index];
    }

    int size() {
      return domain.size();
    }

  };
}

#endif

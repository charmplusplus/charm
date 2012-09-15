#ifndef DATA_ITEM_TYPES_H
#define DATA_ITEM_TYPES_H

template<class dtype, class itype>
class ArrayDataItem{
 public:
  itype arrayIndex;
  dtype dataItem;

  ArrayDataItem(itype i, const dtype d) : arrayIndex(i), dataItem(d) {}
};

#endif

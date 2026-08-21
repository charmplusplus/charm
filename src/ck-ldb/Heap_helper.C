//This has to do with heapify
template <class T, class Compare>
void BubbleDown(int index, std::vector<T>& data, Compare& comp, std::vector<int>& pos) {
    int length = data.size();
    if(length == 0)
        return;
    if(index < 0)
        return;
    int left_child = 2*index + 1;
    int right_child = 2*index + 2;

    if(left_child >= length)
        return; /*index is a leaf*/

    int min_index = index;

    if(comp(data[index], data[left_child])) {
        min_index = left_child;
    }

    if((right_child < length) && comp(data[min_index], data[right_child])) {
        min_index = right_child;
    }

    if(min_index != index) {
    /*need to swap*/
        T temp = data[index];
        data[index] = data[min_index];
        /* Update the position*/
        data[min_index] = temp;
        pos[data[min_index]]=min_index;
        pos[data[index]]=index;
        BubbleDown(min_index, data, comp, pos);
    }
}

template <class T, class Compare>
void BubbleUp(int index, std::vector<T>& data, Compare& comp, std::vector<int>& pos) {
    int length = data.size();
    if(length==0 )
        return;
    if(index <= 0)
        return;

    int parentIndex = (index-1)/2;

    /* The parent is supposed to have smaller value, so swap*/
    if(comp(data[parentIndex], data[index])) {
        T temp = data[parentIndex];
        data[parentIndex] = data[index];

        data[index] = temp;
        /* Update the position*/
        pos[data[parentIndex]] = parentIndex;
        pos[data[index]] = index;
        BubbleUp(parentIndex, data, comp, pos);
    }
}

template <class T, class Compare>
void heap_update(int pe_id, std::vector<T>& data, Compare comp,
    std::vector<int>& pos) {

    int idx = pos[pe_id];
    if (idx < 0) {
        return;
    }
    int parent_index = (idx-1)/2;
    if (idx > 0 && comp(data[parent_index], data[idx])) {
        BubbleUp(idx, data, comp, pos);
    } else {
        BubbleDown(idx, data, comp, pos);
    }
}

template <class T>
void CheckHeapPos(std::vector<T>& data, std::vector<int>& pos) {
    for (int i = 0; i < data.size(); i++) {
        if (pos[data[i]] != i) {
            DEBUG("position of data %d is not %d as expected but is %d\n", data[i], i, pos[data[i]]);
        }
    }
}


template <class T, class Compare>
T heap_pop(std::vector<T>& data, Compare comp, std::vector<int>& pos) {
    int length = data.size();
    if(length==0)
        return -1;
    T ret_val  = data[0];
    if (length != 0) {
        data[0] = data[length-1];
        data.pop_back();
        pos[data[0]]=0;
        BubbleDown(0, data, comp, pos);
    }
    pos[ret_val] = -1;
    return ret_val;
}

template <class T, class Compare>
void heap_insert(std::vector<T>& data, T val, Compare comp, std::vector<int>& pos) {
    int length = data.size();
    data.push_back(val);
    pos[val] = length;
    BubbleUp(length, data, comp, pos);
}

template <class T, class Compare>
void heapify(std::vector<T>& data, Compare comp, std::vector<int>& pos) {
    for (int i = data.size()-1; i >= 0; --i) {
        BubbleDown(i, data, comp, pos);
    }
}

class ObjCompareOperator {
 public:
    ObjCompareOperator(std::vector<CkVertex>* obj, int* gain_val): objs(obj),
      gains(gain_val) {}

    bool operator()(int v1, int v2) {
        //return (gains[v1] < gains[v2]);//wrong
        return (gains[v1] > gains[v2]);//right
    }
 private:
    std::vector<CkVertex>* objs;
    int* gains;
};


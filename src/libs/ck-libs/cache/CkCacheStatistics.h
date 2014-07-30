#ifndef __CACHEMANAGER_STATISTICS_H__
#define __CACHEMANAGER_STATISTICS_H__

class CkCacheStatistics {
  CmiUInt8 dataArrived;
  CmiUInt8 dataTotalArrived;
  CmiUInt8 dataMisses;
  CmiUInt8 dataLocal;
  CmiUInt8 dataError;
  CmiUInt8 totalDataRequested;
  CmiUInt8 maxData;
  int index;

  CkCacheStatistics() : dataArrived(0), dataTotalArrived(0),
    dataMisses(0), dataLocal(0), dataError(0),
    totalDataRequested(0), maxData(0), index(-1) { }

 public:
  CkCacheStatistics(CmiUInt8 pa, CmiUInt8 pta, CmiUInt8 pm,
          CmiUInt8 pl, CmiUInt8 pe, CmiUInt8 tpr,
          CmiUInt8 mp, int i) :
    dataArrived(pa), dataTotalArrived(pta), dataMisses(pm),
    dataLocal(pl), dataError(pe), totalDataRequested(tpr),
    maxData(mp), index(i) { }

  void printTo(CkOStream &os) {
    os << "  Cache: " << dataTotalArrived << " data arrived (corresponding to ";
    os << dataArrived << " messages), " << dataLocal << " from local Chares" << endl;
    if (dataError > 0) {
      os << "Cache: ======>>>> ERROR: " << dataError << " data messages arrived without being requested!! <<<<======" << endl;
    }
    os << "  Cache: " << dataMisses << " misses during computation" << endl;
    os << "  Cache: Maximum of " << maxData << " data stored at a time in processor " << index << endl;
    os << "  Cache: local Chares made " << totalDataRequested << " requests" << endl;
  }

  static CkReduction::reducerType sum;

  static CkReductionMsg *sumFn(int nMsg, CkReductionMsg **msgs) {
    CkCacheStatistics ret;
    ret.maxData = 0;
    for (int i=0; i<nMsg; ++i) {
      CkAssert(msgs[i]->getSize() == sizeof(CkCacheStatistics));
      CkCacheStatistics *data = (CkCacheStatistics *)msgs[i]->getData();
      ret.dataArrived += data->dataArrived;
      ret.dataTotalArrived += data->dataTotalArrived;
      ret.dataMisses += data->dataMisses;
      ret.dataLocal += data->dataLocal;
      ret.totalDataRequested += data->totalDataRequested;
      if (data->maxData > ret.maxData) {
        ret.maxData = data->maxData;
        ret.index = data->index;
      }
    }
    return CkReductionMsg::buildNew(sizeof(CkCacheStatistics), &ret);
  }
};

#endif // __CACHEMANAGER_STATISTICS_H__


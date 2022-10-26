#define PPN_COUNT 64

class Histogram : public CBase_Histogram {
    public:
      inline static int index[PPN_COUNT];
      inline static int *data[PPN_COUNT];
      Histogram();
      Histogram(CkMigrateMessage* msg);
      void insertSend();
      static void userDeliver(int val);
};

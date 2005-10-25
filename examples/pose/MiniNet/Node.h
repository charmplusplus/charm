class nodeMsg {
 public:
  int nbr1, nbr2;
  nodeMsg& operator=(const nodeMsg& obj) {
    eventMsg::operator=(obj);
    nbr1 = obj.nbr1;
    nbr2 = obj.nbr2;
    return *this;
  }
};

class transmitMsg {
 public:
  int src, dest;
  char data[30];
  transmitMsg& operator=(const transmitMsg& obj) {
    eventMsg::operator=(obj);
    src = obj.src;
    dest = obj.dest;
    strcpy(data, obj.data);
    return *this;
  }
};

class Node {
  int nbr1, nbr2;
 public:
  Node();
  Node(nodeMsg *m); 
  ~Node() { }
  Node& operator=(const Node& obj);
  void pup(PUP::er &p) { 
    chpt<state_Node>::pup(p); 
    p(nbr1); p(nbr2);
  }

  // Event methods
  void recv(transmitMsg *m);
  void recv_anti(transmitMsg *m);
  void recv_commit(transmitMsg *m);
};


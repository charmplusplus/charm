#ifndef  _HYBRIDLBMSG_H_
#define  _HYBRIDLBMSG_H_

struct Location {
    LDObjKey key;
    int      loc;
    Location(): loc(0) {}
    Location(const LDObjKey &k, int l): key(k), loc(l) {}
    void pup(PUP::er &p) { p|key; p|loc; }
};

#endif

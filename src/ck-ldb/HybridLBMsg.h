
struct Location {
    LDObjKey key;
    int      loc;
    Location(): loc(0) {}
    Location(LDObjKey &k, int l): key(k), loc(l) {}
    void pup(PUP::er &p) { p|key; p|loc; }
};


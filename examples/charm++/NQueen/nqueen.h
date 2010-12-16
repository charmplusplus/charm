#ifndef _NQueenState_H_
#define _NQueenState_H_

#define MAX 20
#define MAX_BOARDSIZE 30

class QueenState: public CMessage_QueenState {

public:
    int aQueenBitRes;
    int aQueenBitCol;//[MAX_BOARDSIZE];
    int aQueenBitPosDiag;//[MAX_BOARDSIZE];
    int aQueenBitNegDiag;//[MAX_BOARDSIZE];
    
    //int bitfield;
    int numrows;
};

class DUMMYMSG : public CMessage_DUMMYMSG {
public:
};

class NQueen : public CBase_NQueen {

public:

    NQueen(QueenState *);
    NQueen(CkMigrateMessage *msg);
    void sequentialSolve( QueenState *msg);
    
private:
    void printSolution(int[]);

};
#endif

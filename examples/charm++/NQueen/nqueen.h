#ifndef _NQueenState_H_
#define _NQueenState_H_

#define MAX 20
class NQueen : public CBase_NQueen {

public:

    NQueen(QueenState *);
    NQueen(CkMigrateMessage *msg);
    
private:
    //place another queen

    void printSolution(int[]);
    void sequentialSolve(int[], int row, int colum);

    //indicate which colum has a queen for each row
    int current_row;

};

class QueenState: public CMessage_QueenState {

public:
    int row;
    int queenInWhichColum[MAX]; 
};

class DUMMYMSG : public CMessage_DUMMYMSG {
public:
};

#endif

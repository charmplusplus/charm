#ifndef _GLOBALS_H
#define _GLOBALS_H

class Overlapper;
typedef Overlapper * Overlapperp;
typedef Overlapperp * Overlapperpp;
typedef Overlapperpp * Overlapperppp;

CpvDeclare(Overlapperppp , ImplTable);
CpvDeclare(NEWFN*, StrategyTable);
CpvDeclare(int, StrategyTableIndex);
CpvDeclare(int, ImplIndex);
CpvDeclare(int, RecvHandle);
CpvDeclare(int, ProcHandle);
CpvDeclare(int, DummyHandle);
CpvDeclare(int, SwitchHandle);
CpvDeclare(int, KDoneHandle);
CpvDeclare(int, KGMsgHandle);

#endif

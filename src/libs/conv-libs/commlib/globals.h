/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _GLOBALS_H
#define _GLOBALS_H

class Overlapper;
typedef Overlapper * Overlapperp;
typedef Overlapperp * Overlapperpp;
typedef Overlapperpp * Overlapperppp;

CkpvDeclare(Overlapperppp , ImplTable);
CkpvDeclare(NEWFN*, StrategyTable);
CkpvDeclare(int, StrategyTableIndex);
CkpvDeclare(int, ImplIndex);
CkpvDeclare(int, RecvHandle);
CkpvDeclare(int, ProcHandle);
CkpvDeclare(int, DummyHandle);
CkpvDeclare(int, SwitchHandle);
CkpvDeclare(int, KDoneHandle);
CkpvDeclare(int, KGMsgHandle);

#endif

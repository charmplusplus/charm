/**************************************************************************
 * WARNING                                                                *
 **************************************************************************
 * This is a machine generated header file.                               *
 * It is not meant to be edited by hand and may be overwritten by charjc. *
 **************************************************************************/

namespace tests {

#include "Fib.decl.h"
#include <Chare.h>
class Fib : public CBase_Fib  {
    public: Fib* /*variableDeclaratorList*/ parent_;
    public: bool /*variableDeclaratorList*/ root_;
    public: int /*variableDeclaratorList*/ n_;
    public: int /*variableDeclaratorList*/ partialResult_;
    public: int /*variableDeclaratorList*/ pendingChildren_;
    public: Fib( bool /*variableDeclaratorId*/ root,  int /*variableDeclaratorId*/ n,  Fib /*variableDeclaratorId*/ parent) ;
    public: void passUp( int /*variableDeclaratorId*/ subTreeValue) ;
};


} 

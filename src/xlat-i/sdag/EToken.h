/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#ifndef _EToken_H_
#define _EToken_H_

enum EToken {
   DEFAULT=1

  ,IDENT=2
  ,LBRACE=3
  ,RBRACE=4
  ,LB=5
  ,RB=6
  ,LP=7
  ,RP=8
  ,COLON=9
  ,STAR=10
  ,CHAR=11
  ,STRING=12
  ,NEW_LINE=13
  ,CLASS=14
  ,ENTRY=15
  ,SDAGENTRY=16
  ,OVERLAP=17
  ,WHEN=18
  ,IF=19
  ,WHILE=20
  ,FOR=21
  ,FORALL=22
  ,ATOMIC=23
  ,COMMA=24
  ,ELSE=25
  ,SEMICOLON=26
  ,PARAMLIST=27
  ,PARAMETER=28
  ,VARTYPELIST=29
  ,VARTYPE=30
  ,FUNCTYPE=31
  ,SIMPLETYPE=32
  ,BUILTINTYPE=33
  ,ONEPTRTYPE=34
  ,PTRTYPE=35
  ,INT=36
  ,LONG=37
  ,SHORT=38
  ,UNSIGNED=39
  ,DOUBLE=40
  ,VOID=41
  ,FLOAT=42
  ,CONST=43
  ,EQUAL=44
  ,AMPERESIGN=45
  ,LITERAL=46
  ,NUMBER=47
  ,FORWARD=48

  ,MATCHED_CPP_CODE=100
  ,INT_EXPR=101
  ,WSPACE=102
  ,SLIST=103
  ,ELIST=104
  ,OLIST=105
};

#endif /* _EToken_H_ */

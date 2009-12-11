/*
** headerpad.h
** 
** Made by Eric Bohm
** Login   <bohm@alacrity>
** 
** Started on  Thu Dec  3 14:15:51 2009 Eric Bohm
** Last update Thu Dec  3 14:15:51 2009 Eric Bohm
*/

#ifndef   	HEADERPAD_H_
# define   	HEADERPAD_H_
#include "headerpad.decl.h"
class testMsg : public CMessage_testMsg {
public:
  int userdata;
};

class main : public CBase_main
{
public:
  main(CkMigrateMessage *m) {}
  main(CkArgMsg *m);
  void recv(testMsg *m);

};

#endif 	    /* !HEADERPAD_H_ */

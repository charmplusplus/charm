/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

#include "CDep.h"

// called by when function
void CDep::Register(CWhenTrigger *trigger)
{
  whens[trigger->whenID]->append(trigger);
}

// called by entry function
void CDep::deRegister(CWhenTrigger *trigger)
{
  whens[trigger->whenID]->remove(trigger);
}

// called by entry function
void CDep::bufferMessage(int entry, void *msg, int refnum)
{
  CMsgBuffer *buf = new CMsgBuffer(entry, msg, refnum);
  buffers[entry]->append(buf);
  return;
}

// called by when function
CMsgBuffer *CDep::getMessage(int entry)
{
  return buffers[entry]->front();
}

// called by when function
CMsgBuffer *CDep::getMessage(int entry, int refnum)
{
  TListCMsgBuffer *list = buffers[entry];
  for(CMsgBuffer *elem=list->begin(); !list->end(); elem=list->next()) {
    if(elem==0)
      return 0;
    if(elem->refnum == refnum)
      return elem;
  }
  return 0;
}

// called by when function
void CDep::removeMessage(CMsgBuffer *msg)
{
  TListCMsgBuffer *list = buffers[msg->entry];
  list->remove(msg);
}

// called by entry funcion
int CDep::depSatisfied(CWhenTrigger *trigger)
{
  int i;
  for(i=0;i<trigger->nEntries;i++) {
    if(!getMessage(trigger->entries[i], trigger->refnums[i]))
      return 0;
  }
  for(i=0;i<trigger->nAnyEntries;i++) {
    if(!getMessage(trigger->anyEntries[i]))
      return 0;
  }
  return 1;
}

// called by entry function
CWhenTrigger *CDep::getTrigger(int entry, int refnum)
{
  for(int i=0;i<numEntryDepends[entry];i++) {
    TListCWhenTrigger *wlist = entryDepends[entry][i];
    for(CWhenTrigger *elem=wlist->begin(); !wlist->end(); elem=wlist->next()) {
      if(elem==0)
        break;
      if(depSatisfied(elem)){
         deRegister(elem);
         return elem;
      }
    }
  }
  return 0;
}


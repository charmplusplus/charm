#include "barrier_redn.int"
#define ModuleName Barrier
module ModuleName {

message {int size;} REDUCEINIT;

message {ChareNumType id;} BARRIER_MSG;

#define R_BY_MESSAGE  0
#define R_BY_FUNCTION 1


BranchOffice reduce {

    int             flag;
    int             cntval;               /* number of my children + 1    */
    int             cnt;                  /* counter for expected messages*/ 
    int             parent;               /* my parent number */ 
    int             r_type;               /* result return type:by msg or func*/
    int             r_entry;              /* Return Entry Point           */
    void            (* r_function)();     /* result is returned to this func*/ 
    ChareNumType    r_bocnum;             /* boc id of the requester */
    ChareIDType     r_cid;                /* chare id of the requester */
    int             send_result_flag;

    entry init : (message REDUCEINIT *msg)
      { 

          flag = 0;
          cnt = cntval = CkNumSpanTreeChildren(CkMyPe()) + 1;
          parent = CkSpanTreeParent(CkMyPe());
          CkFreeMsg(msg);

      }




    entry collect : (message BARRIER_MSG *msg)
      {
          PrivateCall(send());
          CkFreeMsg(msg);
      }




    entry distribute : (message BARRIER_MSG *msg)
      {

          if ( send_result_flag) 
	  {
             if (r_type == R_BY_FUNCTION) {
                  BranchCall(r_bocnum,r_function(MyBocNum())); 
                  CkFreeMsg(msg);
               } 
             else {
                  msg->id = MyBocNum();
                  SendMsg(r_entry,msg,&r_cid);
                  }
	  }
          else
             CkFreeMsg(msg);
             
      }



    /*  **************************************************************** */
    /*  Service Functions:                                               */
    /*  **************************************************************** */


    /*            returned (return by function call)                       */
    /* ep         entry point where ther result will be sent               */
    /*            (return by message)                                      */
    /* id         pointer to the id of requester boc or chare              */


    /* return by function call */
    public f(fptr,id)
    void     (*fptr)();
    void     *id; 
      {  
          if (flag) CkPrintf("[%d] REDUCTIONLIB : error\n",CkMyPe());
          send_result_flag = 1; 
          r_type = R_BY_FUNCTION;
          if (id == NULL) 
             send_result_flag = 0;
          else
             {
                r_function = fptr;
                r_bocnum = *((ChareNumType *)id);
             }
          flag = 1;
          PrivateCall(send());
      }


    /* return by message */
    public f_msg(ep,id)
    EntryNumType ep;
    void         *id;
       {
          if (flag) CkPrintf("[%d] REDUCTIONLIB:error\n",CkMyPe());
          send_result_flag = 1;
          r_type = R_BY_MESSAGE;
          if (id == NULL) 
             send_result_flag = 0;
          else 
             {
                 r_cid = *((ChareIDType *) id);
                 r_entry = ep;
             }
          flag = 1;
          PrivateCall(send());
       }




    /*  *********************************************************** */
    /*  Internal functions                                          */
    /*  *********************************************************** */

    private send()
      {
          int i;
          BARRIER_MSG *msg;
        
          if (--cnt == 0) {

             /* cnt is zero, all the results from children received and  */
             /* a  request has been issued from this branch              */
             /* therefore, if I am the root,distribute the message, else */
             /* pass the partial result to my parent                     */
             msg = (BARRIER_MSG *) CkAllocMsg(BARRIER_MSG); 
             /*   First reset local variables */
             cnt  = cntval;
             flag = 0;
             if (parent == -1) 
                BroadcastMsgBranch(distribute,msg);
             else
                SendMsgBranch(collect,msg,parent);
          }


      }
    }

 
    Create()
    {
        int        boc;
        REDUCEINIT *msg;

        msg = (REDUCEINIT *) CkAllocMsg(REDUCEINIT);
        boc=CreateBoc(ModuleName::reduce,ModuleName::reduce@init,msg);
        return boc;
    }



    Signal(boc,fptr,id)
    int      boc;
    void     (*fptr)();
    void     *id; 
      {  
          BranchCall(boc,ModuleName::reduce@f(fptr,id));
      }

    /* return by message */
   SignalMsg(boc,ep,id)
    ChareNumType boc; 
    EntryNumType ep;
    void         *id;
       {
          BranchCall(boc,ModuleName::reduce@f_msg(ep,id));
       }




}

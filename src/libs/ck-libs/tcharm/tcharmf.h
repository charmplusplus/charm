       external TCHARM_Set_stack_size
       external TCHARM_Create
       external TCHARM_Create_data
       
       integer, external :: TCHARM_Get_num_chunks
       integer, external :: TCHARM_Element
       integer, external :: TCHARM_Num_elements
       
       external TCHARM_Barrier

       external TCHARM_Register
       external TCHARM_Migrate
       external TCHARM_Migrate_to
       external TCHARM_Yield
       external TCHARM_Done

       external TCHARM_Init

       integer, external :: TCHARM_Iargc
       external TCHARM_Getarg

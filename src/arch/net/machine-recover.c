extern void (*notify_crash_fn)(int);

/**
 * @brief Handles the crash announcement message.
 * For double in-memory checkpoint, it calls the notify crash function.
 */
static void crash_node_handle(ChMessage *m){
	ChMessageInt_t *d = (ChMessageInt_t *)m->data;
	int crashed_node = ChMessageInt(d[0]);
#if CMK_MEM_CHECKPOINT
        if (notify_crash_fn!=NULL) notify_crash_fn(crashed_node);
#endif
	/* tell charmrun we knew */
	ctrl_sendone_nolock("crash_ack",NULL,0,NULL,0);
	// fprintf(stdout,"[%d] got crash mesg for %d \n",CmiMyPe(),crashed_node);
}


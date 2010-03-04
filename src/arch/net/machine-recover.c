extern void notify_crash(int node);

static void crash_node_handle(ChMessage *m){
	ChMessageInt_t *d = (ChMessageInt_t *)m->data;
	int crashed_node = ChMessageInt(d[0]);
#if CMK_MEM_CHECKPOINT
	notify_crash(crashed_node);
#endif
	/* tell charmrun we knew */
	ctrl_sendone_nolock("crash_ack",NULL,0,NULL,0);
	// fprintf(stdout,"[%d] got crash mesg for %d \n",CmiMyPe(),crashed_node);
}


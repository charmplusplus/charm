static int crashed_node;
static void crash_node_handle(ChMessage *m){
	ChMessageInt_t *d = (ChMessageInt_t *)m->data;
	crashed_node = ChMessageInt(d[0]);
	/* tell charmrun we knew */
	ctrl_sendone_nolock("crash_ack",NULL,0,NULL,0);
	// fprintf(stdout,"[%d] got crash mesg for %d \n",CmiMyPe(),crashed_node);
}


/*****************************************************************************
 * $Source$
 * $Author$
 * $Date$
 * $Revision$
 *****************************************************************************/

/* Function declarations for the various socket Routines */


unsigned int skt_ip();
void skt_server(unsigned int *pip,unsigned short *ppo,unsigned int *pfd);
void skt_accept(int src,unsigned int *pip,unsigned int *ppo,unsigned int *pfd);
int skt_connect(unsigned int ip, int port);

class HelloMessage : public ArrayMessage
{
public:
  int hop;

  HelloMessage(void) { CPrintf("hop=0\n"); hop = 0; };
};

Node::Node(nodeMsg *m)
{
  nbr1 = m->nbr1;
  nbr2 = m->nbr2;
  delete m;
  transmitMsg *tm;
  for (int i=0; i<4; i++) { // send to all other nodes
    if (i != myHandle) { // don't send to self
      tm = new transmitMsg;
      tm->src = myHandle;
      tm->dest = i;
      strcpy(tm->data," Hello! ");
      if (nbr1 == i) {
	POSE_invoke(recv(tm), Node, nbr1, 3); // a hop takes 3 time units
      }
      else { // sends to nbr2 or non-nbr always go to nbr2
	POSE_invoke(recv(tm), Node, nbr2, 3); // a hop takes 3 time units
      }
    }
  }
}

Node::Node()
{
}

Node& Node::operator=(const Node& obj)
{
  int i;
  rep::operator=(obj);
  nbr1 = obj.nbr1;
  nbr2 = obj.nbr2;
  return *this;
}

void Node::recv(transmitMsg *m)
{
  if (m->dest == myHandle) { // this message was for me!
    parent->CommitPrintf("%d received %s from %d at time %d\n", 
			 myHandle, m->data, m->src, ovt);
  }
  else { // this message is not for me
    transmitMsg *tm = new transmitMsg;
    tm->src = m->src;
    tm->dest = m->dest;
    strcpy(tm->data, m->data);
    elapse(1); // It took me a time unit to think about how to route this msg
    parent->CommitPrintf("%d forwarding message from %d to %d at time %d\n", 
			 myHandle, m->src, m->dest, ovt);
    if (nbr1 == tm->dest) { // sophisticated routing algorithm!
      POSE_invoke(recv(tm), Node, nbr1, 3); // a hop takes 3 time units
    }
    else {
      POSE_invoke(recv(tm), Node, nbr2, 3); // a hop takes 3 time units
    }
  }
}

void Node::recv_anti(transmitMsg *m)
{
  restore(this);
}

void Node::recv_commit(transmitMsg *m)
{
}

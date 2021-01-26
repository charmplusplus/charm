#!/usr/bin/env python3
import os

cores_per_node = 42
n_sockets = 2
procs_per_socket = 3

job_id = os.environ['LSB_JOBID']
lsb_hosts = os.environ['LSB_HOSTS']
x = lsb_hosts.split(' ')[1:]
hosts = [x[i] for i in range(len(x)) if i % cores_per_node == 0]
n_hosts = len(hosts)
#print('Hosts:', hosts)
#print('# of hosts:', n_hosts)

rankfile = open("rankfile-" + job_id, "w")

lines = ''
rank = 0
for host in hosts:
  for socket in range(n_sockets):
    for proc in range(procs_per_socket):
      line = 'rank ' + str(rank) + '=' + host + ' slot=' + str(socket) + ':' + str(proc) + '\n'
      lines += line
      rank += 1

#print(lines)
rankfile.write(lines)

rankfile.close()

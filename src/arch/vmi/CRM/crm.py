#!/usr/local/bin/python
###########################################################################
## Greg Koenig (koenig@uiuc.edu)
##
## This program is the Charm Resource Manager (CRM), designed to allow
## portions of a Charm program distributed across multiple clusters in a
## Grid computing environment to discover each other.  The independent
## pieces of the overall job may be started separately on different
## clusters.  Each process in each independent piece contacts the CRM,
## specifying a common "program key" and the expected total number of
## processes in the overall job.  The CRM waits until the expected number
## of processes check in with the same program key and then arranges the
## processes in increasing order of TCP/IP addresses.  The list of
## processes is then returned to each process in the overall computation.
## This allows each process to know the ranks of every process in the
## computation.  Further setup (such as opening VMI connections) takes
## place after this.
##
## This program is designed specifically to work with programs launched
## by the companion program charmgrid.py.  It is similar in many ways
## to charmrun from the Charm net-linux version, but designed to
## work in Grid environments where independent pieces of a Grid job
## must be started separately.  For the single cluster case, either
## charmrun or charmgrid can be used to start jobs.  (The Charm
## vmi-linux version also understands the charmrun protocol.)
##
## This code requires Python 2.3.x or higher!

import os
import select
import socket
import string
import struct
import sys
import time

# These are constants that can be changed to configure the program.
CRM_PORT        = 7777
CRM_TIMEOUT     = 300
CRM_LOGFILENAME = '/home/koenig/crm.log'

# These are constants that SHOULD NOT be changed.
CRM_MSG_SUCCESS  = 0
CRM_MSG_REGISTER = 1
CRM_MSG_FAILED   = 3

CRM_ERR_CTXTCONFLICT = 1
CRM_ERR_TIMEOUT      = 3



###########################################################################
## 
def write_to_log (log_str):
    try:
        crm_logfile = open (CRM_LOGFILENAME, 'a')
        crm_logfile.write ('[%s] %s\n' % (time.ctime(time.time()), log_str))
        crm_logfile.close ()

    except:
        pass



###########################################################################
## This function gets the registration data sent from a process checking
## in with the CRM.  This data is:
##
##    message identifier (4 bytes - int)
##    total number of processes in Grid computation (4 bytes - int)
##    TCP/IP port that the process is listening on (4 bytes - int)
##    length of program key (4 bytes - int)
##    program key (N bytes - char)
##
## The function returns a list of the data sent by the client.
##
def socket_getdata (s):
    try:
        packed_code = s.recv (4)
        packed_np   = s.recv (4)
        packed_port = s.recv (4)
        packed_len  = s.recv (4)
        packed_data = packed_code + packed_np + packed_port + packed_len

        unpacked_data = struct.unpack ('iiii', packed_data)

        msg_code = socket.ntohl (unpacked_data[0])
        msg_np   = socket.ntohl (unpacked_data[1])
        msg_port = socket.ntohl (unpacked_data[2])
        msg_len  = socket.ntohl (unpacked_data[3])

        msg_key = s.recv (msg_len)

        return ([msg_code, msg_np, msg_port, msg_key])

    except:
        return ([])



###########################################################################
## This is the main program.
##
## Two primary data structures are used in this program:
##
##    * unaffiliated connections - a list of all processes that have
##      connected to the CRM but have not associated themselves with
##      a particular program key; each entry is represented by a list
##      containing its information:
##
##      [TCP/IP address (int), socket, checkin time (seconds since epoch)]
##
##    * affiliated connections - a list of all processes that have
##      connected to the CRM and sent their registration data, and are
##      now waiting for the CRM to coordinate their peers; each entry
##      is represented by a list containing its information:
##
##      [program key, number of processes, last process checkin
##       time (seconds since epoch), [TCP/IP address (int), socket, port]]
##
def main():
    write_to_log ('CRM starting')

    unaffiliated = []
    affiliated = []

    listen_socket = socket.socket (socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.setsockopt (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind ( ('', CRM_PORT) )
    listen_socket.listen (5)

    while True:
        # Get the current time (seconds past epoch) for this iteration.
        time_now = int (time.time ())

        # Check to see if there is a new incoming connection.
        a1, a2, a3 = select.select ([listen_socket], [], [], 0.1)
	if (len (a1) > 0):
            in_socket, in_address = listen_socket.accept ()
            in_address_list = string.split (in_address[0], '.')
            in_address_int = int (in_address_list[0]) * 2**24 + \
                             int (in_address_list[1]) * 2**16 + \
                             int (in_address_list[2]) * 2**8  + \
                             int (in_address_list[3])
            a = [in_address_int, in_socket, time_now]
            unaffiliated.append (a)
            write_to_log ('New connection from ' + in_address[0])

        # Drop any unaffiliated connections that have timed out.
	for i in unaffiliated:
            unaff_ip, unaff_socket, unaff_time = i
            if (time_now > (unaff_time + CRM_TIMEOUT)):
                msg_code = socket.htonl (CRM_MSG_FAILED)
                msg_error = socket.htonl (CRM_ERR_TIMEOUT)
                packed_data = struct.pack ('ii', msg_code, msg_error)
                unaff_socket.send (packed_data)
                unaff_socket.close ()
                unaffiliated.remove (i)
                write_to_log ('Unaffiliated connection timed out')

        # Get input from any unaffiliated connections that have sent data.
        # Based on the program key, move the connection to the affiliated list.
        a = []
        for i in unaffiliated:
            unaff_ip, unaff_socket, unaff_time = i
            a.append (unaff_socket)
        a1, a2, a3 = select.select (a, [], [], 0.1)
        for i in a1:
            # a1 is a list of all sockets that have data.
            # i is an iterator through this list.
            parsed_data = socket_getdata (i)
            if (len (parsed_data) > 0):
                msg_code, msg_np, msg_port, msg_key = parsed_data
                if (msg_code == CRM_MSG_REGISTER):
                    # This socket returned a registration code.
                    aff_key = ''
                    for j in affiliated:
                        aff_key, aff_np, aff_time, aff_ip_sock_port = j
                        if (msg_key == aff_key):
                            if (msg_np == aff_np):
                                # Gave right key and number of processes.
                                # Update affiliated list entry with current time.
                                # Remove this connection from unaffiliated list.
                                j[2] = time_now
                                for k in unaffiliated:
                                    if (k[1] == i):
                                        unaff_ip, unaff_socket, unaff_time = k
                                        break
                                j[3].append ( [unaff_ip,i,msg_port] )
                                unaffiliated.remove (k)
                                write_to_log ('Unaffiliated connection joined key ' + msg_key)
                                break
                            else:
                                # Gave right key, but wrong number of processes.
                                # Send a failure code to the connection.
                                # Remove the connection from the unaffiliated list.
                                for j in unaffiliated:
                                    unaff_ip, unaff_socket, unaff_time = j
                                    if (unaff_socket == i):
                                        msg_code = socket.htonl (CRM_MSG_FAILED)
                                        msg_error = socket.htonl (CRM_ERR_CTXTCONFLICT)
                                        packed_data = struct.pack ('ii', msg_code, msg_error)
                                        unaff_socket.send (packed_data)
                                        unaff_socket.close ()
                                        unaffiliated.remove (j)
                                        write_to_log ('Unaffiliated connection gave wrong number of processors for key ' + msg_key)
                                        break
                                break
                    if (msg_key != aff_key):
                        # Did not find the key in the entire affiliated list.
                        # This means a new key needs to be started.
                        for j in unaffiliated:
                            unaff_ip, unaff_socket, unaff_time = j
                            if (unaff_socket == i):
                                break
                        new_entry = [msg_key, msg_np, time_now, [[unaff_ip,i,msg_port]]]
                        affiliated.append (new_entry)
                        unaffiliated.remove (j)
                        write_to_log ('Unaffiliated connection started new key ' + msg_key + ' for ' + str(msg_np) + ' processors')
                else:
                    # This socket sent a bad message code - dump it.
                    for j in unaffiliated:
                        unaff_ip, unaff_socket, unaff_time = j
                        if (unaff_socket == i):
                            unaff_socket.close ()
                            unaffiliated.remove (j)
                            write_to_log ('Unaffiliated connection sent bad message code - dropped')
                            break
            else:
                # This socket sent bad (zero-length?) data - dump it.
                for j in unaffiliated:
                    unaff_ip, unaff_socket, unaff_time = j
                    if (unaff_socket == i):
                        unaff_socket.close ()
                        unaffiliated.remove (j)
                        write_to_log ('Unaffiliated connection sent bad data - dropped')
                        break

        # Drop any affiliated connections that have timed out.
	for i in affiliated:
            aff_key, aff_np, aff_time, aff_ip_sock_port = i
            if (time_now > (aff_time + CRM_TIMEOUT)):
                for j in aff_ip_sock_port:
                    aff_ip, aff_socket, aff_port = j
                    msg_code = socket.htonl (CRM_MSG_FAILED)
                    msg_error = socket.htonl (CRM_ERR_TIMEOUT)
                    packed_data = struct.pack ('ii', msg_code, msg_error)
                    aff_socket.send (packed_data)
                    aff_socket.close ()
                affiliated.remove (i)
                write_to_log ('Coordination for key ' + aff_key + ' timed out - all connections dropped')

        # Drop any affiliated connections that send bad data.
        # These connections shouldn't actually send ANY data.
        # If they do send data, just dump it unless it is
        # zero-length data (indicating the client probably
        # disappeared).  The other affiliated connections for
        # a given key remain intact.  Any key that has all its
        # affiliated connections disappear is removed from the
        # affiliated list and discarded.
        a = []
        for i in affiliated:
            aff_key, aff_np, aff_time, aff_ip_sock_port = i
            for j in aff_ip_sock_port:
                aff_ip, aff_socket, aff_port = j
                a.append (aff_socket)
        a1, a2, a3 = select.select (a, [], [], 0.1)
        for i in a1:
            # a1 is a list of all sockets that have data.
            # i is an iterator through this list.
            in_data = i.recv (1024)
            if (len (in_data) == 0):
                # Drop this connection - it sent zero-length data.
                # This usually means that the remote process disappeared.
                for j in affiliated:
                    aff2_key, aff2_np, aff2_time, aff2_ip_sock_port = j
                    for k in aff2_ip_sock_port:
                        aff2_ip, aff2_socket, aff2_port = k
                        if (aff2_socket == i):
                            break
                    aff2_socket.close ()
                    j[3].remove (k)
                    write_to_log ('Affiliated connection on key ' + aff2_key + ' sent bad data - dropped')
                    if (len (j[3]) == 0):
                        affiliated.remove (j)
                        write_to_log ('Key ' + aff2_key + ' is now empty - discarded')

        # Respond to any members of the affiliated list that have had all
        # processes in the group check in.
        for i in affiliated:
            aff_key, aff_np, aff_time, aff_ip_sock_port = i
            if (aff_np == len (aff_ip_sock_port)):
                # Sort the list of [TCP/IP address, socket, port] connections.
                # Since TCP/IP address is first, the sort is on this field.
                aff_ip_sock_port.sort ()

                # Iterate through each connection and send a success code
                # to the connection followed by the list of all processes
                # in the key.  After this, close the connection.
                for j in aff_ip_sock_port:
                    aff_ip, aff_socket, aff_port = j

                    msg_code = socket.htonl (CRM_MSG_SUCCESS)
                    msg_np = socket.htonl (aff_np)
                    packed_data = struct.pack ('ii', msg_code, msg_np)
                    aff_socket.send (packed_data)

                    tmp = 0
                    for k in aff_ip_sock_port:
                        tmp_ip, tmp_sock, tmp_port = k

                        tmp_ip = socket.htonl (tmp_ip)
                        tmp_port = socket.htonl (tmp_port)
                        tmp = socket.htonl (tmp)

                        packed_data = struct.pack ('Lii', tmp_ip,tmp_port,tmp)

                        aff_socket.send (packed_data)

                        tmp = tmp + 1

                    aff_socket.close ()

                affiliated.remove (i)
                write_to_log ('Key ' + aff_key + ' synchronization complete')



main ()

#!/usr/bin/python
###########################################################################
## Greg Koenig (koenig@uiuc.edu)
##
## This program is the charmgrid client, designed to launch a Charm program
## on the cluster nodes allocated to a job.  It is very similar to charmrun
## from the Charm net-linux version, but designed specifically to work with
## the Charm Resource Manager (CRM) to launch independent pieces of a Grid
## job on multiple clusters in a Grid environment.
##
## This code requires Python 2.3.x or higher!

import os
import string
import sys

# These are constants for default values that can be changed by the user.
try:
    HOME_DIRECTORY = os.environ['HOME']

    DEFAULT_NODELIST_FILENAME = HOME_DIRECTORY + '/.nodelist'
    DEFAULT_NODEGROUP         = 'main'
    DEFAULT_VMI_PROCS         = '1'
    DEFAULT_CRM               = '10.92.0.253'
    DEFAULT_VMI_KEY           = 'gak'
    DEFAULT_VMI_SPECFILE      = '/home/koenig/ON-DEMAND/VMI22-install/specfiles/myrinet.xml'

except KeyError:
    print 'ERROR: Unable to get home directory from environment.'
    sys.exit (1)



###########################################################################
##
def ParseCommandLine ():
    # Set default values for all variables.
    nodelist_filename                = DEFAULT_NODELIST_FILENAME
    nodegroup                        = DEFAULT_NODEGROUP
    vmi_procs                        = DEFAULT_VMI_PROCS
    crm                              = DEFAULT_CRM
    vmi_key                          = DEFAULT_VMI_KEY
    vmi_specfile                     = DEFAULT_VMI_SPECFILE
    verbose                          = False
    #
    vmi_gridprocs                    = ''
    wan_latency                      = ''
    cluster_number                   = ''
    probe_clusters                   = ''
    grid_queue                       = ''
    grid_queue_maximum               = ''
    grid_queue_interval              = ''
    grid_queue_threshold             = ''
    #
    memory_pool                      = ''
    connection_timeout               = ''
    maximum_handles                  = ''
    small_message_boundary           = ''
    medium_message_boundary          = ''
    eager_protocol                   = ''
    eager_interval                   = ''
    eager_threshold                  = ''
    eager_short_pollset_size_maximum = ''
    eager_short_slots                = ''
    eager_long_buffers               = ''
    eager_long_buffer_size           = ''
    disable_regcache                 = False
    ##
    command                          = ''

    # If the user didn't give any command-line arguments, display usage and exit.
    if (len (sys.argv) == 1):
        DisplayUsage ()
        sys.exit (1)

    # Parse the command line.
    i = 1
    j = len (sys.argv)
    while (i < j):
        arg = sys.argv[i]
        i = i + 1

        if (arg == '++nodelist'):
            nodelist_filename = sys.argv[i]
            i = i + 1
        elif (arg == '++nodegroup'):
            nodegroup = sys.argv[i]
            i = i + 1
	elif (arg == '++p'):
            vmi_procs = sys.argv[i]
            i = i + 1
        elif (arg == '++crm'):
            crm = sys.argv[i]
            i = i + 1
        elif (arg == '++key'):
            vmi_key = sys.argv[i]
            i = i + 1
        elif (arg == '++specfile'):
            vmi_specfile = sys.argv[i]
            i = i + 1
        elif (arg == '++verbose'):
            verbose = True
        elif (arg == '++help'):
            DisplayUsage ()
            sys.exit (0)
        #
	elif (arg == '++g'):
            vmi_gridprocs = sys.argv[i]
            i = i + 1
	elif (arg == '++wan-latency'):
            wan_latency = sys.argv[i]
            i = i + 1
	elif (arg == '++cluster'):
            cluster_number = sys.argv[i]
            i = i + 1
	elif (arg == '++probe-clusters'):
            probe_clusters = sys.argv[i]
            i = i + 1
        elif (arg == '++grid-queue'):
            grid_queue = sys.argv[i]
            i = i + 1
        elif (arg == '++grid-queue-maximum'):
            grid_queue_maximum = sys.argv[i]
            i = i + 1
        elif (arg == '++grid-queue-interval'):
            grid_queue_interval = sys.argv[i]
            i = i + 1
        elif (arg == '++grid-queue-threshold'):
            grid_queue_threshold = sys.argv[i]
            i = i + 1
        #
        elif (arg == '++memory-pool'):
            memory_pool = sys.argv[i]
            i = i + 1
	elif (arg == '++connection-timeout'):
            connection_timeout = sys.argv[i]
            i = i + 1
	elif (arg == '++maximum-handles'):
            maximum_handles = sys.argv[i]
            i = i + 1
	elif (arg == '++small-message-boundary'):
            small_message_boundary = sys.argv[i]
            i = i + 1
	elif (arg == '++medium-message-boundary'):
            medium_message_boundary = sys.argv[i]
            i = i + 1
        elif (arg == '++eager-protocol'):
            eager_protocol = sys.argv[i]
            i = i + 1
	elif (arg == '++eager-interval'):
            eager_interval = sys.argv[i]
            i = i + 1
	elif (arg == '++eager-threshold'):
            eager_threshold = sys.argv[i]
            i = i + 1
	elif (arg == '++eager-short-pollset-size-maximum'):
            eager_short_pollset_size_maximum = sys.argv[i]
            i = i + 1
	elif (arg == '++eager-short-slots'):
            eager_short_slots = sys.argv[i]
            i = i + 1
	elif (arg == '++eager-long-buffers'):
            eager_long_buffers = sys.argv[i]
            i = i + 1
	elif (arg == '++eager-long-buffer-size'):
            eager_long_buffer_size = sys.argv[i]
            i = i + 1
        elif (arg == '++disable-regcache'):
            disable_regcache = True
        ##
        else:
            if (arg[0:2] == '++'):
                DisplayUsage ()
                sys.exit (1)
            else:
                command = arg
                while (i < j):
                    command = command + ' ' + sys.argv[i]
                    i = i + 1

    # If the user did not specify a number of processors for a Grid
    # job, then the number defaults to the number of processors in
    # this single job (i.e., there is only a single subjob for this job).
    if (vmi_gridprocs == ''):
        vmi_gridprocs = vmi_procs

    # Return a list of all variables.
    return [nodelist_filename, nodegroup, vmi_procs, crm, vmi_key, vmi_specfile, verbose,                                                          \
            vmi_gridprocs, wan_latency, cluster_number, probe_clusters, grid_queue, grid_queue_maximum, grid_queue_interval, grid_queue_threshold, \
            memory_pool, connection_timeout, maximum_handles, small_message_boundary, medium_message_boundary, eager_protocol, eager_interval,     \
            eager_threshold, eager_short_pollset_size_maximum, eager_short_slots, eager_long_buffers, eager_long_buffer_size, disable_regcache, command]



###########################################################################
##
def DisplayUsage ():
    print ' '
    print 'USAGE: charmgrid [options] <command> [command arguments]'
    print ' '
    print 'common options include:'
    print '  ++nodelist <nodelist>     the Charm nodelist to use'
    print '  ++nodegroup <nodegroup>   the Charm nodegroup to use'
    print '  ++p <#>                   the number of processes in this job (or subjob)'
    print '  ++crm <CRM>               the Charm Resource Manager (CRM) to coordinate with'
    print '  ++key <key>               the program key to use for coordination with CRM'
    print '  ++specfile <specfile>     the VMI device specfile to use'
    print '  ++verbose                 displays additional information during startup'
    print '  ++help                    displays this help text'
    print ' '
    print 'Grid options include:'
    print '  ++g <#>                                  total number of processes in entire'
    print '                                           Grid job (combined ++p subjobs must'
    print '                                           add up to ++g job size)'
    print '  ++wan-latency <latency>                  inter-node latencies below this'
    print '                                           indicate nodes on the same LAN;'
    print '                                           above means WAN (microseconds)'
    print '  ++cluster <#>                            the cluster number for all processes'
    print '                                           in this subjob'
    print '  ++probe-clusters <0|1>                   disable/enable automatic probing the'
    print '                                           cluster topology of the job'
    print '  ++grid-queue <0|1>                       disable/enable the Grid queue which'
    print '                                           allows prioritization of Grid objects'
    print '  ++grid-queue-maximum <count>             maximum number of Grid border objects'
    print '                                           per processor'
    print '  ++grid-queue-interval <#>                after every # message sends, look for'
    print '                                           Grid object candidates'
    print '  ++grid-queue-threshold <#>               a sender must send # Grid messages'
    print '                                           over Grid interval to be a Grid'
    print '                                           object candidate'
    print ' '
    print 'tuning options include:'
    print '  ++memory-pool <0|1>                      disable/enable memory pool'
    print '  ++connection-timeout <seconds>           timeout to wait for connection setup'
    print '  ++maximum-handles <#>                    number of send/receive handles'
    print '                                           (grows automatically as program runs)'
    print '  ++small-message-boundary <bytes>         messages below boundary use small'
    print '                                           protocol (inline stream or eager Put)'
    print '  ++medium-message-boundary <bytes>        messages below boundary use medium'
    print '                                           protocol (stream or eager Put)'
    print '  ++eager-protocol <0|1>                   disable/enable eager protocol'
    print '  ++eager-interval <#>                     after every # message receives, look'
    print '                                           for eager protocol candidates'
    print '  ++eager-threshold <#>                    a sender must send # messages over'
    print '                                           eager interval to be eager candidate'
    print '  ++eager-short-pollset-size-maximum <#>   maximum number of processes that can'
    print '                                           be switched to eager short protocol'
    print '  ++eager-short-slots <#>                  eager short buffer gets broken into'
    print '                                           this many slots (window count)'
    print '  ++eager-long-buffers <#>                 number of eager long buffers to set'
    print '                                           up (window count)'
    print '  ++eager-long-buffer-size <bytes>         maximum size of an eager long message'
    print '  ++disable-regcache                       disable the VMI cache manager'
    print '                                           (significantly decreases performance)'
    print ' '



###########################################################################
## Parse the nodelist file.  This is in the charmrun nodelist format which
## is documented in the Charm++ Installation and Usage manual.
##
## This function returns a list of node information:
##
##    [[group name, [node1, node2, ...]], ...]
##
def ParseNodelistFile (nodelist):
    infile = open (nodelist, 'r')
    inlines = infile.readlines ()
    infile.close ()

    for i in range (len (inlines)):
        inlines[i] = string.strip (inlines[i])

    nodes = []
    
    i = 0
    j = len (inlines)
    while (i < j):
        line = inlines[i]
        i = i + 1

        if (line[0] == '#'):
            # Any line beginning with a '#' is a comment.
            pass
        elif (line[0:5] == 'group'):
            # Found a new group list.
            # First get the group's name.
            group_name = line[6:]
            
            # Next get the hosts in the group.
            host_list = []
            cont = True
            while ((i < j) and cont):
                line = inlines[i]
                i = i + 1

                if (line[0] == '#'):
                    # Any line beginning with a '#' is a comment.
                    pass
                elif (line[0:4] == 'host'):
                    # Found a host in the group.
                    host_name = line[5:]
                    host_list.append (host_name)
                else:
                    # Read past the last host in the group - back up.
                    i = i - 1
                    cont = False

            # Add this group into the node list.
            group = [group_name, host_list]
            nodes.append (group)
        else:
            print 'Malformed nodelist file'
            sys.exit (1)
            
    return nodes



###########################################################################
## Write a script to launch the job portion on a single node.
##
def WriteNodeScript (filename, crm, vmi_key, vmi_specfile, vmi_gridprocs, wan_latency, cluster_number, probe_clusters, grid_queue, grid_queue_maximum, \
                     grid_queue_interval, grid_queue_threshold, memory_pool, connection_timeout, maximum_handles, small_message_boundary,              \
                     medium_message_boundary, eager_protocol, eager_interval, eager_threshold, eager_short_pollset_size_maximum, eager_short_slots,    \
                     eager_long_buffers, eager_long_buffer_size, disable_regcache, working_directory, command):

    outfile = open (filename, 'w')

    ##

    outfile.write ('CRM="' + crm + '"')
    outfile.write (' ; export CRM\n')

    outfile.write ('VMI_KEY="' + vmi_key + '"')
    outfile.write (' ; export VMI_KEY\n')

    outfile.write ('VMI_SPECFILE="' + vmi_specfile + '"')
    outfile.write (' ; export VMI_SPECFILE\n')

    outfile.write ('VMI_PROCS="' + vmi_gridprocs + '"')
    outfile.write (' ; export VMI_PROCS\n')

    outfile.write ('VMI_MMAP_MAX="0" ; export VMI_MMAP_MAX\n')

    #

    if (wan_latency != ''):
        outfile.write ('CMI_VMI_WAN_LATENCY="' + wan_latency + '"')
        outfile.write (' ; export CMI_VMI_WAN_LATENCY\n')

    if (cluster_number != ''):
        outfile.write ('CMI_VMI_CLUSTER="' + cluster_number + '"')
        outfile.write (' ; export CMI_VMI_CLUSTER\n')

    if (probe_clusters != ''):
        outfile.write ('CMI_VMI_PROBE_CLUSTERS="' + probe_clusters + '"')
        outfile.write (' ; export CMI_VMI_PROBE_CLUSTERS\n')

    if (grid_queue != ''):
        outfile.write ('CMI_VMI_GRID_QUEUE="' + grid_queue + '"')
        outfile.write (' ; export CMI_VMI_GRID_QUEUE\n')

    if (grid_queue_maximum != ''):
        outfile.write ('CMI_VMI_GRID_QUEUE_MAXIMUM="' + grid_queue_maximum + '"')
        outfile.write (' ; export CMI_VMI_GRID_QUEUE_MAXIMUM\n')

    if (grid_queue_interval != ''):
        outfile.write ('CMI_VMI_GRID_QUEUE_INTERVAL="' + grid_queue_interval + '"')
        outfile.write (' ; export CMI_VMI_GRID_QUEUE_INTERVAL\n')

    if (grid_queue_threshold != ''):
        outfile.write ('CMI_VMI_GRID_QUEUE_THRESHOLD="' + grid_queue_threshold + '"')
        outfile.write (' ; export CMI_VMI_GRID_QUEUE_THRESHOLD\n')

    #

    if (memory_pool != ''):
        outfile.write ('CMI_VMI_MEMORY_POOL="' + memory_pool + '"')
        outfile.write (' ; export CMI_VMI_MEMORY_POOL\n')
        
    if (connection_timeout != ''):
        outfile.write ('CMI_VMI_CONNECTION_TIMEOUT="' + connection_timeout + '"')
        outfile.write (' ; export CMI_VMI_CONNECTION_TIMEOUT\n')

    if (maximum_handles != ''):
        outfile.write ('CMI_VMI_MAXIMUM_HANDLES="' + maximum_handles + '"')
        outfile.write (' ; export CMI_VMI_MAXIMUM_HANDLES\n')

    if (small_message_boundary != ''):
        outfile.write ('CMI_VMI_SMALL_MESSAGE_BOUNDARY="' + small_message_boundary + '"')
        outfile.write (' ; export CMI_VMI_SMALL_MESSAGE_BOUNDARY\n')

    if (medium_message_boundary != ''):
        outfile.write ('CMI_VMI_MEDIUM_MESSAGE_BOUNDARY="' + medium_message_boundary + '"')
        outfile.write (' ; export CMI_VMI_MEDIUM_MESSAGE_BOUNDARY\n')

    if (eager_protocol != ''):
        outfile.write ('CMI_VMI_EAGER_PROTOCOL="' + eager_protocol + '"')
        outfile.write (' ; export CMI_VMI_EAGER_PROTOCOL\n')
        
    if (eager_interval != ''):
        outfile.write ('CMI_VMI_EAGER_INTERVAL="' + eager_interval + '"')
        outfile.write (' ; export CMI_VMI_EAGER_INTERVAL\n')

    if (eager_threshold != ''):
        outfile.write ('CMI_VMI_EAGER_THRESHOLD="' + eager_threshold + '"')
        outfile.write (' ; export CMI_VMI_EAGER_THRESHOLD\n')

    if (eager_short_pollset_size_maximum != ''):
        outfile.write ('CMI_VMI_EAGER_SHORT_POLLSET_SIZE_MAXIMUM="' + eager_short_pollset_size_maximum + '"')
        outfile.write (' ; export CMI_VMI_EAGER_SHORT_POLLSET_SIZE_MAXIMUM\n')

    if (eager_short_slots != ''):
        outfile.write ('CMI_VMI_EAGER_SHORT_SLOTS="' + eager_short_slots + '"')
        outfile.write (' ; export CMI_VMI_EAGER_SHORT_SLOTS\n')

    if (eager_long_buffers != ''):
        outfile.write ('CMI_VMI_EAGER_LONG_BUFFERS="' + eager_long_buffers + '"')
        outfile.write (' ; export CMI_VMI_EAGER_LONG_BUFFERS\n')

    if (eager_long_buffer_size != ''):
        outfile.write ('CMI_VMI_EAGER_LONG_BUFFER_SIZE="' + eager_long_buffer_size + '"')
        outfile.write (' ; export CMI_VMI_EAGER_LONG_BUFFER_SIZE\n')

    if (disable_regcache):
        outfile.write ('VMI_DISABLE_REGCACHE="1"')
        outfile.write (' ; export VMI_DISABLE_REGCACHE\n')

    ##

    outfile.write ('cd ' + working_directory + '\n')

    outfile.write (command + '\n')

    outfile.close ()



###########################################################################
## Launch job_script on node.  To do this, first fork.  The child process
## opens job_script and then unlinks it (so it is deleted after the child
## closes it and/or ends).  The child then sets its stdin to the job_script
## file descriptor and executes /usr/bin/ssh to node to launch the job.
##
def LaunchJob (node, job_script):
    pid = os.fork ()
    if (pid == 0):
        job_input = open (job_script, 'r')
        os.unlink (job_script)
        os.dup2 (job_input.fileno (), sys.stdin.fileno ())
        os.execv ('/usr/bin/ssh', ('/usr/bin/ssh', '-x', node, '/bin/sh'))
        exit (0)
    else:
        return pid



###########################################################################
## This is the main program.
##
def main ():
    # Parse the command line.
    parsed_command_line = ParseCommandLine ()

    nodelist_filename                = parsed_command_line[0]
    nodegroup                        = parsed_command_line[1]
    vmi_procs                        = parsed_command_line[2]
    crm                              = parsed_command_line[3]
    vmi_key                          = parsed_command_line[4]
    vmi_specfile                     = parsed_command_line[5]
    verbose                          = parsed_command_line[6]
    #
    vmi_gridprocs                    = parsed_command_line[7]
    wan_latency                      = parsed_command_line[8]
    cluster_number                   = parsed_command_line[9]
    probe_clusters                   = parsed_command_line[10]
    grid_queue                       = parsed_command_line[11]
    grid_queue_maximum               = parsed_command_line[12]
    grid_queue_interval              = parsed_command_line[13]
    grid_queue_threshold             = parsed_command_line[14]
    #
    memory_pool                      = parsed_command_line[15]
    connection_timeout               = parsed_command_line[16]
    maximum_handles                  = parsed_command_line[17]
    small_message_boundary           = parsed_command_line[18]
    medium_message_boundary          = parsed_command_line[19]
    eager_protocol                   = parsed_command_line[20]
    eager_interval                   = parsed_command_line[21]
    eager_threshold                  = parsed_command_line[22]
    eager_short_pollset_size_maximum = parsed_command_line[23]
    eager_short_slots                = parsed_command_line[24]
    eager_long_buffers               = parsed_command_line[25]
    eager_long_buffer_size           = parsed_command_line[26]
    disable_regcache                 = parsed_command_line[27]
    ##
    command                          = parsed_command_line[28]

    # Parse the nodelist file.
    # Locate the group name and its list of nodes that corresponds to
    # the group specified by the user (default to "main").
    parsed_nodelist = ParseNodelistFile (nodelist_filename)

    for i in range (len (parsed_nodelist)):
        group_name = parsed_nodelist[i][0]
        nodelist = parsed_nodelist[i][1]

        if (group_name == nodegroup):
            break

    if (group_name != nodegroup):
        print 'The specified nodegroup was not found in the nodelist file.'
        sys.exit (1)

    # Get this processes process ID and current working directory.
    pid = os.getpid ()
    cwd = os.getcwd ()

    # Figure out the command to execute to launch the job and the correct
    # working directory to use.
    if (command[0] == '/'):
        # The command begins with an absolute path (possibly followed by
        # a space and arguments).
        i = string.find (command, ' ')
        cmd = command[0:i]
        args = command[i:]

        # Break the command containing an absolute path and program name
        # into a path component (which becomes the working directory)
        # and a command (+ arguments) component.
        cmdpath,cmdprog = os.path.split (cmd)
    
        working_directory = cmdpath
        command = './' + cmdprog + args
    elif (command[0:2] == './'):
        # The command begins with a './' which indicates it uses a
        # relative path.  The working directory is the current working
        # directory.
        working_directory = cwd
    else:
        # The command begins with just the name of the command, so we
        # append a './' to it.  The working directory is the current
        # working directory.
        working_directory = cwd
        command = './' + command

    # If the working directory begins with the user's home directory,
    # modify it so it uses the symbol $HOME instead.  For example,
    # /home/koenig/programdirectory/command is modified to be
    # $HOME/programdirectory/command instead.
    #
    # The reason for this modification is that programs can be launched
    # on different machines in a Grid environment even if individual
    # clusters use a different path for user home directories, as long
    # as the user maintains the same relative path under their home
    # directory on each machine.
    if (working_directory[0:len(HOME_DIRECTORY)] == HOME_DIRECTORY):
        working_directory2 = '$HOME' + working_directory[len(HOME_DIRECTORY):]
        working_directory = working_directory2

    # Manipulate the path to the VMI specfile in the same manner as
    # the working directory above.
    if (vmi_specfile[0:len(HOME_DIRECTORY)] == HOME_DIRECTORY):
        vmi_specfile2 = '$HOME' + vmi_specfile[len(HOME_DIRECTORY):]
        vmi_specfile = vmi_specfile2

    # Launch each process in the job and wait for them to complete.
    child_pids = []
    try:
        j = 0
        for i in range (int (vmi_procs)):
            # Write a job script used to launch process i.
            # This script is /tmp/charmgrid.pid.i
            # The child unlinks (deletes) this script during launch.
            job_script = '/tmp/charmgrid.' + str (pid) + '.' + str (i)
            WriteNodeScript (job_script, crm, vmi_key, vmi_specfile, vmi_gridprocs, wan_latency, cluster_number, probe_clusters, grid_queue,    \
                             grid_queue_maximum, grid_queue_interval, grid_queue_threshold, memory_pool, connection_timeout, maximum_handles,   \
                             small_message_boundary, medium_message_boundary, eager_protocol, eager_interval, eager_threshold,                  \
                             eager_short_pollset_size_maximum, eager_short_slots, eager_long_buffers, eager_long_buffer_size, disable_regcache, \
                             working_directory, command)

            # Get the node to launch on.  This is round-robin across
            # the nodes in the group specified by the user.
            node = nodelist[j]
            j = j + 1
            if (j >= len (nodelist)):
                j = 0

            # Launch the job on the node.  Store the child process ID
            # so we can wait on it below.
            child_pid = LaunchJob (node, job_script)
            child_pids.append (child_pid)

        # Wait for all child processes launched to complete.
        while (len (child_pids) > 0):
            done_pid,rc = os.wait ()
            child_pids.remove (done_pid)

    except:
        # There was some error, so kill all children.
        for i in child_pids:
            os.kill (i, 1)   # signal 1 is SIGHUP



main ()

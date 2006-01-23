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
    DEFAULT_NODELIST     = HOME_DIRECTORY + '/.nodelist'
    DEFAULT_NODEGROUP    = 'main'
    DEFAULT_CRM          = '141.142.222.53'   # koenig.ncsa.uiuc.edu
    DEFAULT_VMI_PROCS    = '1'
    DEFAULT_VMI_SPECFILE = '/home/koenig/ON-DEMAND/VMI21-install/specfiles/myrinet.xml'
    DEFAULT_VMI_KEY      = 'gak'
    DEFAULT_VERBOSE      = False
    DEFAULT_COMMAND      = ''

except KeyError:
    print 'ERROR: Unable to get home directory from environment.'
    sys.exit (1)



###########################################################################
##
def ParseCommandLine ():
    # Set default values for all variables.
    nodelist         = DEFAULT_NODELIST
    nodegroup        = DEFAULT_NODEGROUP
    crm              = DEFAULT_CRM
    vmi_procs        = DEFAULT_VMI_PROCS
    vmi_specfile     = DEFAULT_VMI_SPECFILE
    vmi_key          = DEFAULT_VMI_KEY
    verbose          = DEFAULT_VERBOSE
    command          = DEFAULT_COMMAND
    vmi_gridprocs    = -1
    disable_regcache = False

    # If they didn't give any command-line arguments, display usage and exit.
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
            nodelist = sys.argv[i]
            i = i + 1

        elif (arg == '++nodegroup'):
            nodegroup = sys.argv[i]
            i = i + 1

        elif (arg == '++crm'):
            crm = sys.argv[i]
            i = i + 1

	elif (arg == '++p'):
            vmi_procs = sys.argv[i]
            i = i + 1

	elif (arg == '++g'):
            vmi_gridprocs = sys.argv[i]
            i = i + 1

        elif (arg == '++specfile'):
            vmi_specfile = sys.argv[i]
            i = i + 1

        elif (arg == '++key'):
            vmi_key = sys.argv[i]
            i = i + 1

        elif (arg == '++disable-regcache'):
            disable_regcache = True

        elif (arg == '++verbose'):
            verbose = True

        elif (arg == '++help'):
            DisplayUsage ()
            sys.exit (0)

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
    if (vmi_gridprocs <= 0):
        vmi_gridprocs = vmi_procs

    # Return a list of all variables.
    return [nodelist, nodegroup, crm, vmi_procs, vmi_gridprocs, \
            vmi_specfile, vmi_key, disable_regcache, verbose, command]



###########################################################################
##
def DisplayUsage ():
    print ' '
    print 'USAGE: charmgrid [options] <command> [command arguments]'
    print ' '
    print '[options] include:'
    print '  ++nodelist           the Charm++ nodelist to use'
    print '  ++nodegroup          the Charm++ nodegroup to use (default="main")'
    print '  ++crm                the Charm++ Resource Manager to coordinate with'
    print '  ++p                  the number of processors to start for this sub-job'
    print '  ++g                  the total number of processors in an entire Grid job'
    print '                       (defaults to g=p if not specified)'
    print '  ++specfile           the VMI specfile to use'
    print '  ++key                the program key to use for coordination with CRM'
    print '  ++disable-regcache   disable VMI cache manager'
    print '                       (this option significantly decreases performance)'
    print '  ++verbose            displays additional information during startup'
    print '  ++help               displays this help text'
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
def WriteNodeScript (filename, crm, vmi_specfile, vmi_gridprocs, vmi_key, \
                     disable_regcache, working_directory, command):
    outfile = open (filename, 'w')

    outfile.write ('CRM="' + crm + '"')
    outfile.write (' ; export CRM\n')

    outfile.write ('VMI_SPECFILE="' + vmi_specfile + '"')
    outfile.write (' ; export VMI_SPECFILE\n')

    outfile.write ('VMI_PROCS="' + vmi_gridprocs + '"')
    outfile.write (' ; export VMI_PROCS\n')

    outfile.write ('VMI_KEY="' + vmi_key + '"')
    outfile.write (' ; export VMI_KEY\n')

    if (disable_regcache):
        outfile.write ('VMI_DISABLE_REGCACHE="1"')
        outfile.write (' ; export VMI_DISABLE_REGCACHE\n')

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

    nodelist_filename = parsed_command_line[0]
    nodegroup         = parsed_command_line[1]
    crm               = parsed_command_line[2]
    vmi_procs         = parsed_command_line[3]
    vmi_gridprocs     = parsed_command_line[4]
    vmi_specfile      = parsed_command_line[5]
    vmi_key           = parsed_command_line[6]
    disable_vmicache  = parsed_command_line[7]
    verbose           = parsed_command_line[8]
    command           = parsed_command_line[9]

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
            WriteNodeScript (job_script, crm, vmi_specfile, vmi_gridprocs, \
                             vmi_key, disable_regcache, working_directory, \
                             command)

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

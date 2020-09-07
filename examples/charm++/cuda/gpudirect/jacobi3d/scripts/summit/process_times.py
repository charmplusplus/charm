#!/usr/bin/env python3
import os
import sys
import csv
import math
import statistics

# Number of GPUs
gpu_count = 6

# Number of jobs per config
job_count = 3

# Number of runs per job
run_count = 3

# Number of ODFs
odf_count = 5

count_per_job = run_count * odf_count

if len(sys.argv) != 4:
  print('Please use', sys.argv[0], '[job name] [start node count] [end node count]')
  exit()

job_name = sys.argv[1]
start_node_count = int(sys.argv[2])
end_node_count = int(sys.argv[3])

csv_filename = job_name + '.csv'
csv_file = open(csv_filename, 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['Number of GPUs', 'Charm-ODF-1', 'error', 'Charm-ODF-2', 'error', 'Charm-ODF-4', 'error', 'Charm-ODF-8', 'error', 'Charm-ODF-16', 'error'])

node_count_list = []
cur_node_count = start_node_count
while cur_node_count <= end_node_count:
  node_count_list.append(cur_node_count)
  cur_node_count *= 2

for node_count in node_count_list:
  print('Node count:', str(node_count))
  comm_str = 'grep -ir "average iteration time" ' + job_name + '-n' + str(node_count) + '.* | cut -d " " -f4'

  stream = os.popen(comm_str)
  lines = stream.readlines()
  timelist = list(map(lambda x: x / 1000, list(map(float, list(map(str.rstrip, lines))))))

  odf_1_times = []
  odf_2_times = []
  odf_4_times = []
  odf_8_times = []
  odf_16_times = []

  for jobi in range(0,job_count):
    job_timelist = timelist[jobi*count_per_job:(jobi+1)*count_per_job]
    for odfi in range(0,odf_count):
      odf_timelist = job_timelist[odfi*run_count:(odfi+1)*run_count]
      if odfi == 0:
        odf_1_times += odf_timelist
      elif odfi == 1:
        odf_2_times += odf_timelist
      elif odfi == 2:
        odf_4_times += odf_timelist
      elif odfi == 3:
        odf_8_times += odf_timelist
      elif odfi == 4:
        odf_16_times += odf_timelist
  print(odf_1_times)
  print(odf_2_times)
  print(odf_4_times)
  print(odf_8_times)
  print(odf_16_times)

  odf_1_avg = statistics.mean(odf_1_times)
  odf_1_stdev = statistics.stdev(odf_1_times)
  #odf_1_error = 1.96 * odf_1_stdev / math.sqrt(len(odf_1_times))
  odf_2_avg = statistics.mean(odf_2_times)
  odf_2_stdev = statistics.stdev(odf_2_times)
  odf_4_avg = statistics.mean(odf_4_times)
  odf_4_stdev = statistics.stdev(odf_4_times)
  odf_8_avg = statistics.mean(odf_8_times)
  odf_8_stdev = statistics.stdev(odf_8_times)
  odf_16_avg = statistics.mean(odf_16_times)
  odf_16_stdev = statistics.stdev(odf_16_times)

  writer.writerow([str(node_count * gpu_count) + "\\n(" + str(node_count) + ")", str(odf_1_avg), str(odf_1_stdev), str(odf_2_avg), str(odf_2_stdev), str(odf_4_avg), str(odf_4_stdev), str(odf_8_avg), str(odf_8_stdev), str(odf_16_avg), str(odf_16_stdev)])

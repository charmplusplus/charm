set title "Simulation data for Processors"
set xlabel "Time Intervals"
set ylabel "Percentage Utilization"
plot "sample-out2.txt" using 1:2 with linespoints lt 5
set out 'sample-graph.ps'
load 'saveplot'

#!/bin/sh
# Shell script to test for multiple test cases

touch outputs
for rows1 in 1000 5000 10000; do
  for cols1 in 500 750 1000; do
    for cols2 in 1000 5000 10000; do
      for mbytes in 128 64 32 16 8 4 2 1; do
	for num_workers in 1 2 4 8 16 32; do
          rm -rf params.h
          printf "unsigned int bytes = %d*1024*1024;\n" $mbytes >> params.h
          printf "unsigned int ROWS1 = %d;\n" $rows1 >> params.h
          printf "unsigned int COLS1 = %d;\n" $cols1 >> params.h
          printf "unsigned int COLS2 = %d;\n" $cols2 >> params.h
          printf "unsigned int ROWS2 = COLS1;\n" >> params.h
          printf "unsigned int NUM_WORKERS = %d;\n" $num_workers >> params.h
          printf "\n" >> params.h
  
          rm -f t2d
          make OPTS=-O3 -s
          for num_pes in 4 8 16 32; do
            ./charmrun t2d +p$num_pes >> outputs
          done
        done
      done
    done
  done
done

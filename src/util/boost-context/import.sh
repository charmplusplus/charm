#!/bin/bash

for i in *_ms_pe_gas.asm ; do
  mv -f "$i" "${i%.asm}.S"
done

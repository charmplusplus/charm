#!/bin/bash
# Run this when updating boost-context.

for i in *_ms_pe_gas.asm ; do
  mv -f "$i" "${i%.asm}.S"
done

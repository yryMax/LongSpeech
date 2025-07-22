#!/bin/bash
for file in "$1"/*.tar.gz; do

  dirname=$(basename "$file" .tar.gz)
  
  mkdir -p "$1/$dirname"
  
  tar -xzvf "$file" -C "$1/$dirname"
done
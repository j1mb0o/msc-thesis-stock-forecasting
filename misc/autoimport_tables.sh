#!/bin/bash
#
for file in ../tables/1d/train-restricted-years/*.tex; do echo "\input{$file}"; done
echo ""

for file in ../tables/1d/train-restricted-years-pct/*.tex; do echo "\input{$file}"; done


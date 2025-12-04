#!/bin/bash
#
for timefreq in 1d 1h; do
    for subfolder in ../tables/$timefreq/*/; do
        echo ""
        echo "% $timefreq - $(basename "$subfolder")"
        for file in "$subfolder"*.tex; do
            if [ -f "$file" ]; then
                echo "\input{$file}"
            fi
        done
    done
done


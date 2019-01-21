#!/usr/bin/env bash

set mxct=0
set lgfile=""
for x in $(ls -1 low_dim/)
do
    set ct=$(head -1 "low_dim/$x" | awk -F ',' '{print NF}')
    echo "File: " $x
    echo "Count: " $ct
    if [ $ct -gt $mxct ]; then
        set mxct=$ct
        set lgfile=$x
    fi
done
echo $mxct
echo $lgfile

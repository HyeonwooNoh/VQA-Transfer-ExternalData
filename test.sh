#!/bin/sh
for cuda in 0 1; do
    if [ "$cuda" -eq "0" ]; then
        dp='True'
    else
        dp='False'
    fi
    echo $cuda, $dp
done

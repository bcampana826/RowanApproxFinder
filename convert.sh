#!/bin/bash
# $1 is labels, $2 is edges, $3 is out file

if [[ $# -ne 3 ]]; then
	echo "faulty input. valid: labels, edge list, out file"
	exit -1
fi

# note >
echo -n "t 1 " > $3

wc -l < $1 >> $3


python map_labels.py $1 >> $3

awk '{ print "e " $1, $2}' $2 >> $3

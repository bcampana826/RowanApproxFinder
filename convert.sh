#!/bin/bash
# $1 is labels, $2 is edges, $3 is out file

if [[ $# -ne 3 ]]; then
	echo "faulty input. valid: labels, edgelist, out_file"
	exit -1
fi

echo -n "t 1 " > $3

wc -l < $1 >> $3

awk '{print "v " $1, $2}' $1 >> $3

awk '{print "e " $1, $2}' $2 >> $3

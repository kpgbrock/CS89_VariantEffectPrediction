#!/bin/sh
#PBS -v PATH
#$ -v PATH


para=$1
cd /data5/data/webcomp/web-session/1658330861
./1658330861.fas.0.3866-bl.pl 0 $para &
./1658330861.fas.0.3866-bl.pl 1 $para &
./1658330861.fas.0.3866-bl.pl 2 $para &
./1658330861.fas.0.3866-bl.pl 3 $para &
wait


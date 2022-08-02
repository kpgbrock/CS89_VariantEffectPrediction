#!/bin/sh
#$ -S /bin/bash
#$ -v PATH=/home/data/webcomp/RAMMCAP-ann/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#$ -v BLASTMAT=/home/data/webcomp/RAMMCAP-ann/blast/bin/data
#$ -v LD_LIBRARY_PATH=/home/data/webcomp/RAMMCAP-ann/gnuplot-install/lib
#$ -v PERL5LIB=/home/hying/programs/Perl_Lib
#$ -q cdhit_webserver.q,fast.q
#$ -pe orte 4
#$ -l h_rt=24:00:00


#$ -e /data5/data/webcomp/web-session/1658330861/1658330861.err
#$ -o /data5/data/webcomp/web-session/1658330861/1658330861.out
cd /data5/data/webcomp/web-session/1658330861
sed -i "s/\x0d/\n/g" 1658330861.fas.0

faa_stat.pl 1658330861.fas.0

/data5/data/NGS-ann-project/apps/cd-hit/psi-cd-hit/psi-cd-hit.pl -i 1658330861.fas.0 -o 1658330861.fas.1 -c 0.15 -P /data5/data/NGS-ann-project/apps/blast+/bin -para 4
rm -rf 1658330861.fas.0-bl
faa_stat.pl 1658330861.fas.1
/data5/data/NGS-ann-project/apps/cd-hit/clstr_sort_by.pl no < 1658330861.fas.1.clstr > 1658330861.fas.1.clstr.sorted
/data5/data/NGS-ann-project/apps/cd-hit/clstr_list.pl 1658330861.fas.1.clstr 1658330861.clstr.dump
gnuplot1.pl < 1658330861.fas.1.clstr > 1658330861.fas.1.clstr.1; gnuplot2.pl 1658330861.fas.1.clstr.1 1658330861.fas.1.clstr.1.png
/data5/data/NGS-ann-project/apps/cd-hit/clstr_list_sort.pl 1658330861.clstr.dump 1658330861.clstr_no.dump
/data5/data/NGS-ann-project/apps/cd-hit/clstr_list_sort.pl 1658330861.clstr.dump 1658330861.clstr_len.dump len
/data5/data/NGS-ann-project/apps/cd-hit/clstr_list_sort.pl 1658330861.clstr.dump 1658330861.clstr_des.dump des
tar -zcf 1658330861.result.tar.gz * --exclude=*.dump --exclude=*.env
echo hello > 1658330861.ok

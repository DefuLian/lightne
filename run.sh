#!/bin/bash
dim=128
network_name='network'
label_name='group'
max_iter=50
gamma=0
ratio=1
rank=256
T=10
b=1


usage()
{
  echo "Usage: run [-T=window -b=negative -d=dim --rank=rank 
            --nname=network_name --lname=label_name  
		    --ratio=ratio --max_iter= --gamma=gamma] input"
  exit 2
}

PARSED_ARGUMENTS=$(getopt -a -n run -o T:b:d: --long nname:,lname:,max_iter:,gamma:,ratio:,rank: -- "$@")
VALID_ARGUMENTS=$?
if [ "$VALID_ARGUMENTS" != "0" ]; then
  usage
fi

eval set -- "$PARSED_ARGUMENTS"
while :
do
  case "$1" in
    -T)       T="$2"           ; shift 2 ;;
    -b)       b="$2"           ; shift 2 ;;
    -d)       dim="$2"         ; shift 2 ;;
    --nname)  network_name="$2"; shift 2 ;;
	--lname)  label_name="$2"  ; shift 2 ;;
	--max_iter) max_iter="$2"  ; shift 2 ;;
	--gamma)  gamma="$2"       ; shift 2 ;;
	--ratio)  ratio="$2"       ; shift 2 ;;
	--rank)   rank="$2"        ; shift 2 ;;
    # -- means the end of the arguments; drop this, and break out of the while loop
    --) shift; break ;;
    # If invalid options were passed, then getopt should have reported an error,
    # which we checked as VALID_ARGUMENTS when getopt was called...
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done
if [ $# -eq 0 ]; then
    usage
fi

echo "T            : $T"
echo "b            : $b "
echo "dim          : $dim"
echo "network_name : $network_name"
echo "label_name   : $label_name"
echo "max_iter     : $max_iter"
echo "gamma        : $gamma"
echo "ratio        : $ratio"
echo "rank         : $rank"
echo "Parameters remaining are: $1"
input=$1
input_dir=$(dirname "${input}")
output=$input_dir/embedding_2020.txt
matlab -nosplash -nodesktop -r "run_hash('$input', '$output', 'nn', '$network_name', 'T', $T, 'b', $b, 'dim', $dim, 'ratio', $ratio, 'gamma', $gamma, 'max_iter', $max_iter, 'rank', $rank); exit"
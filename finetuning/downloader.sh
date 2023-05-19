#!/bin/bash
while getopts i:o:s:e: flag
do
    case "${flag}" in
        i) input=${OPTARG};;
        o) output=${OPTARG};;
        s) s=${OPTARG};;
        e) e=${OPTARG};;
    esac
done
prefix="https://data.commoncrawl.org/"
for ((i=$s; i <= $e; i++))
do
    dl_path=`sed -n "$i"p "$input"`
    cmd=`wget -P $output $prefix$dl_path`
    my_ar=($(echo $dl_path | tr "/" "\n")) 
    file_name=${my_ar[5]}
    unzip=`gzip -d $output$file_name`
done
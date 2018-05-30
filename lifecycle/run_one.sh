#!/bin/bash
source ~/.bash_profile
set -x

cmd="run_one param_baseline" 

if [ $# -ge 1 ]; then 
    cmd=$1
fi
function failandexit(){
    if [ $1 -ne 0 ];then
        exit $1
    fi
}
function run_one(){
#    python main.py &
    set -x -e
    param_name=$1
    dat=dat
    mkdir -p $dat/$param_name
    rm -rf $dat/$param_name/*
    cp model/model* $dat/$param_name/
    python main.py --param-file $param_name 2>&1 | tee log.$param_name
    failandexit $?
    eventfile=$(ls -ltr $dat/$param_name/event* | sed -n '$p' | awk '{print $9}')
    echo $eventfile 
    python extract.py $dat/$param_name $dat/$param_name $eventfile 
    cd $dat
    tar -czvf $param_name".tar.gz" $param_name/*.txt && ossput.sh $param_name".tar.gz"  
    cd ..
}
function run_one_new(){
    set -x -e
    param=$1
    dat=$2
    src=$3
    mkdir -p $dat/$param
    rm -rf $dat/$param/*
    cp $src/model/model* $dat/$param/
    cp $src/$param $dat/$param/
    cd $dat/$param/
    python $src/main.py --param-file $dat/$param/$param 2>&1 | tee log.$param
    failandexit $?
    eventfile=$(ls -ltr event* | sed -n '$p' | awk '{print $9}')
    echo $eventfile 
    python $src/extract.py . . $eventfile 
    tar -czvf $param".tar.gz" *.txt 
    cd $src
}
function run_exp(){
#    python main.py &
    set -x -e
    param_name=$1
    dat=dat
    mkdir -p $dat/$param_name
    rm -rf $dat/$param_name/*
    cp model/model* $dat/$param_name/
    python main.py --param-file $param_name --batch-size 320 2>&1 | tee log.$param_name
    failandexit $?
    eventfile=$(ls -ltr $dat/$param_name/event* | sed -n '$p' | awk '{print $9}')
    echo $eventfile 
    python extract.py $dat/$param_name $dat/$param_name $eventfile 
    cd $dat
    tar -czvf $param_name".tar.gz" $param_name/*.txt && ossput.sh $param_name".tar.gz"  
    cd ..
}

function put_one(){
    param_name=$1
    dat=dat
    eventfile=$(ls -ltr $dat/$param_name/event* | sed -n '$p' | awk '{print $9}')
    echo $eventfile 
    python extract.py $dat/$param_name $dat/$param_name $eventfile 
    cd $dat
    tar -czvf $param_name".tar.gz" $param_name/*.txt && ossput.sh $param_name".tar.gz"  
    cd ..
}    
    
function wait_list(){
    for pid in $1
    do
        wait pid
    done
}

function run_noise(){
    python ou_noise.py
    ossput.sh dat/noise.txt
}

$cmd

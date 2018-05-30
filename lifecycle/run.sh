#!/bin/bash
source ~/.bash_profile
set -x

cmd="run_one param_baseline" 
bizdate=$(date -d"now" +%Y%m%d)

if [ $# -ge 1 ]; then 
    cmd=$1
fi

function run_reward(){
    nohup sh run_one.sh "run_one param_high_dim1d" &
    nohup sh run_one.sh "run_one param_high_dim1d_02" &
    nohup sh run_one.sh "run_one param_high_dim1d_03" &
    nohup sh run_one.sh "run_one param_high_dim1d_04" &
}

function run_many(){
    params=("param_T_Perm" "param_FPC" "param_FPC_CNN" "param_FPC_CNN_EXP")
#    params=("param_FPC_CNN_EXP")
    rootpath=`pwd`
    ordmin=14
    ordmax=34
    ord=$ordmin
    while [ 0 -le 1 ]; do
        if [ $ord -ge $ordmax ]; then
            break
        fi
        ord=`expr $ord + 1`
        dat=$rootpath/dat/`printf "E%03d" $ord`
        src=$rootpath
        for par in ${params[@]}; do
#             nohup sh run_one.sh "run_one_new $par $dat $src" &
             nohup sh run_one.sh "run_one_new $par $dat $src" &
#             echo `printf "%03d" $ord` $par $dat $src
        done
    done  
}

function run_collect(){
    #params=("param_T_Perm" "param_FPC" "param_FPC_CNN" "param_FPC_CNN_EXP")
    params=("param_FPC_CNN_EXP")
    dat=./dat
    trg=./dat/trg
    filename=reward_avg.txt
    LINECOUNT=128750
    ord=0
    maxord=13
    rm -rf $trg
    mkdir -p $trg
    while [ 0 -le 1 ]; do
        ord=`expr $ord + 1`
        if [ $ord -ge $maxord ]; then
            break
        fi
        expname=`printf "E%03d" $ord`
        expdir=$dat/$expname
        if [ -d $expdir ]; then
            for parname in ${params[@]}; do
                paradir=$expdir/$parname
                filedir=$expdir/$parname/$filename
                lines=`cat $filedir | wc -l`
                echo $filedir
                if [ -f $filedir ]  && [ $lines -ge $LINECOUNT ]; then
                     targetdir=$trg/$expname/$parname/
                     mkdir -p $targetdir
                     cp $filedir $targetdir
                     echo $targetdir/$filename
                     sh run_odps.sh "upload $bizdate $expname $parname $filedir"
                fi
            done
        fi
    done
    
    tar -czvf $trg".tar.gz" $trg && ossput.sh $trg."tar.gz"
}

function run_analyse(){
    params=("param_T_Perm" "param_FPC" "param_FPC_CNN" "param_FPC_CNN_EXP")
    for par in ${params[@]}; do
        sh run_odps.sh "analyse_param $bizdate $par"
        sh run_odps.sh "download $bizdate avg $par"
    done 
    sh run_odps.sh "putall $bizdate avg"
}

function run_all(){
    nohup sh run_one.sh "run_one param_low_dim1d" &
    nohup sh run_one.sh "run_one param_high_dim1d" &
    nohup sh run_one.sh "run_one param_cnn_dim2d" &
    nohup sh run_one.sh "run_one param_exp_cnn" &
    nohup sh run_one.sh "run_one param_pca_expcnn" &
}
function put_all(){
    nohup sh run_one.sh "put_one param_low_dim1d" &
    nohup sh run_one.sh "put_one param_high_dim1d" &
    nohup sh run_one.sh "put_one param_cnn_dim2d" &
    nohup sh run_one.sh "put_one param_exp_cnn" &
    nohup sh run_one.sh "put_one param_pca_expcnn" &
}
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

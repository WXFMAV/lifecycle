#!/bin/bash
source ~/.bash_profile
#set -x

cmd=put2cloud

if [ $# -ge 1 ]; then 
    cmd=$1
fi
function wait_list(){
    for pid in $1
    do
        wait pid
    done
}
function run_main(){
#    python main.py &
    ef_low_dim1d=$(ls -ltr dat/param_low_dim1d/event* | sed -n '$p' | awk '{print $9}')
    ef_high_dim1d=$(ls -ltr dat/param_high_dim1d/event* | sed -n '$p' | awk '{print $9}')
    ef_cnn_dim2d=$(ls -ltr dat/param_cnn_dim2d/event* | sed -n '$p' | awk '{print $9}')
    echo $ef_low_dim1d 
    echo $ef_high_dim1d
    echo $ef_cnn_dim2d
    pl=""
    python extract.py dat/param_low_dim1d dat/param_low_dim1d $ef_low_dim1d  &
    pl=$pl" $!"
    python extract.py dat/param_high_dim1d dat/param_high_dim1d $ef_high_dim1d & 
    pl=$pl" $!"
    python extract.py dat/param_cnn_dim2d dat/param_cnn_dim2d $ef_cnn_dim2d &
    pl=$pl" $!"
    wait_list $pl
}

function put2cloud(){
    cd dat
    pl=""
    tar -czvf param_low_dim1d_data.tar.gz param_low_dim1d/*.txt && ossput.sh param_low_dim1d_data.tar.gz & 
    pl=$pl" $!"
    tar -czvf param_high_dim1d_data.tar.gz param_high_dim1d/*.txt && ossput.sh param_high_dim1d_data.tar.gz &
    pl=$pl" $!"
    tar -czvf param_cnn_dim2d_data.tar.gz param_cnn_dim2d/*.txt && ossput.sh param_cnn_dim2d_data.tar.gz &
    pl=$pl" $!"
    wati_list $pl
    cd ..
}

function run_one(){
    param_name=param_baseline
    
}
$cmd

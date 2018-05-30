#!/bin/sh

source ~/.bash_profile

#cmd=get_from_cloud
cmd="get_from_cloud_one param_baseline"

if [ $# -ge 1 ]; then
    cmd=$1
fi

function get_from_cloud(){
    cd dat
    ossget.sh param_low_dim1d.tar.gz && tar -xzvf param_low_dim1d.tar.gz
    ossget.sh param_high_dim1d.tar.gz && tar -xzvf param_high_dim1d.tar.gz
    ossget.sh param_cnn_dim2d.tar.gz && tar -xzvf param_cnn_dim2d.tar.gz
    ossget.sh param_exp_cnn.tar.gz && tar -xzvf param_exp_cnn.tar.gz
    cd ..
}
function get_from_cloud_one(){
    param_name=$1
    cd dat
    ossget.sh $param_name".tar.gz" && tar -xzvf $param_name".tar.gz"
    cd ..
}
$cmd

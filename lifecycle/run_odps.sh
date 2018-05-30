#!/bin/sh
source ~/.bash_profile
set -x -e

cmd="upload 20180323 E001 param_FPC dat/E001/param_FPC/reward_avg.txt"
echo $#
if [ $# -ge 1 ]; then
    cmd=$1
fi

function upload(){
    bizdate=$1
    expname=$2
    parname=$3
    filedir=$4
    tablename=lc_reward_all
    odpscmd -e "
        create table if not exists $tablename
        (
            step bigint
            ,reward double
        )
        partitioned by
        (
            ds string
            ,expname string
            ,parname string
        )
        lifecycle 90
        ;
    "
    odpscmd -e "
        alter table $tablename drop if exists partition(ds='$bizdate', expname='$expname', parname='$parname');
        alter table $tablename add if not exists partition(ds='$bizdate', expname='$expname', parname='$parname');
    "
    dship upload $filedir $tablename/ds="$bizdate",expname="$expname",parname="$parname" -fd ' '  
}

function analyse_param(){
    bizdate=$1
    parname=$2
    tablename=lc_reward_all
    odpscmd -e "
        insert overwrite table $tablename partition(ds='$bizdate', expname='avg', parname='$parname')
        select step, avg(double(reward)) as reward
        from $tablename
        where ds='$bizdate'
            and parname='$parname'
            and expname like 'E%'
        group by step
        order by bigint(step) limit 1000000
        ;
    "
    odpscmd -e "
        insert overwrite table $tablename partition(ds='$bizdate', expname='stddev', parname='$parname')
        select step, stddev(double(reward)) as reward
        from $tablename
        where ds='$bizdate'
            and parname='$parname'
            and expname like 'E%'
        group by step
        order by bigint(step) limit 1000000
        ;
    "
    odpscmd -e "
        insert overwrite table $tablename partition(ds='$bizdate', expname='max', parname='$parname')
        select step, max(double(reward)) as reward
        from $tablename
        where ds='$bizdate'
            and parname='$parname'
            and expname like 'E%'
        group by step
        order by bigint(step) limit 1000000
        ;
    "
    odpscmd -e "
        insert overwrite table $tablename partition(ds='$bizdate', expname='min', parname='$parname')
        select step, min(double(reward)) as reward
        from $tablename
        where ds='$bizdate'
            and parname='$parname'
            and expname like 'E%'
        group by step
        order by bigint(step) limit 1000000 
        ;
    "
    odpscmd -e "
        insert overwrite table $tablename partition(ds='$bizdate', expname='count', parname='$parname')
        select step, count(double(reward)) as reward
        from $tablename
        where ds='$bizdate'
            and parname='$parname'
            and expname like 'E%'
        group by step
        order by bigint(step) limit 1000000
        ;
    "
}

function download(){
    bizdate=$1
    expname=$2
    parname=$3
    tablename=lc_reward_all
    dat=dat
    file=$dat/$tablename.$bizdate.$expname.$parname".txt"
    dship download $tablename/ds=$bizdate,expname=$expname,parname=$parname $file -fd '\t'   
#    cd $dat
#    tar -czvf $tablename.$bizdate.$expname.$parname".tar.gz" $tablename.$bizdate.$expname.$parname".txt" && ossput.sh $tablename.$bizdate.$expname.$parname".tar.gz" 
#    cd -
}
function putall(){
    bizdate=$1
    expname=$2
    tablename=lc_reward_all
    dat=dat
    tarname=$expname
    cd $dat
    tar -czvf $tarname".tar.gz" $tablename.$bizdate.$expname".*.txt" && ossput.sh $tarname".tar.gz"
    cd -  
}

function dship_all(){
#    dship download $tablename/ds=$bizdate,expname=$expname,parname=$parname $file -fd '\t'   
    dship download lc_reward_all/ds=20180323,expname=avg,parname=param_T_Perm lc_reward_all.20180323.avg.param_T_Perm.txt -fd '\t'
    dship download lc_reward_all/ds=20180323,expname=avg,parname=param_FPC lc_reward_all.20180323.avg.param_FPC.txt -fd '\t'
    dship download lc_reward_all/ds=20180323,expname=avg,parname=param_FPC_CNN lc_reward_all.20180323.avg.param_FPC_CNN.txt -fd '\t'
    dship download lc_reward_all/ds=20180402,expname=avg,parname=param_FPC_CNN_EXP lc_reward_all.20180323.avg.param_FPC_CNN_EXP.txt -fd '\t'

    dship download lc_reward_all/ds=20180323,expname=count,parname=param_T_Perm lc_reward_all.20180323.count.param_T_Perm.txt -fd '\t'
    dship download lc_reward_all/ds=20180323,expname=count,parname=param_FPC lc_reward_all.20180323.count.param_FPC.txt -fd '\t'
    dship download lc_reward_all/ds=20180323,expname=count,parname=param_FPC_CNN lc_reward_all.20180323.count.param_FPC_CNN.txt -fd '\t'
    dship download lc_reward_all/ds=20180402,expname=count,parname=param_FPC_CNN_EXP lc_reward_all.20180323.count.param_FPC_CNN_EXP.txt -fd '\t'
    tar -czvf avg.tar.gz lc_reward_all.20180323.*.txt && ossput.sh avg.tar.gz
}
$cmd

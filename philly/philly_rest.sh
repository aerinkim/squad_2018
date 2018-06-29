#!/usr/bin/env bash
###############################
# Philly utilities
# by yuwfan@microsoft.com
###############################


USERNAME="yuwfan"
CLUSTER="rr1"
VC="pnrsy"

submit() {
    local CMD="https://philly/api/submit?"
    NAME=$1

    # config and data file
    CMD+="configFile=yuwfan/mrc/san_mrc/run_philly.sh&"
    CMD+="inputDir=/hdfs/pnrsy/yuwfan/mrc/data&"

    # parameters
    # context128_lr002_drop04_contextshare_contextmaxout
    CMD+="extraParams=--name+$1+--contextual_hidden_size+128+--learning_rate+0.002+--dropout_p+0.4+--contextual_encoder_share+--contextual_maxout_on&"

    # other configurations
    CMD+="buildId=10568&"
    CMD+="tag=pytorch0.3.1-py36-msgpack&"
    CMD+="toolType=cust&"
    CMD+="clusterId=$CLUSTER&"
    CMD+="vcId=$VC&"
    CMD+="minGPUs=1&"
    CMD+="name=$NAME&"
    CMD+="isdebug=false&"
    CMD+="iscrossrack=false&"
    CMD+="Repository=philly/jobs/test/pytorch&"
    CMD+="userName=$USERNAME"

    echo $CMD
}

abort() {
    local CMD="https://philly/api/abort?"
    JOB_ID=$1
    CMD+="clusterId=$CLUSTER&"
    CMD+="jobId=$JOB_ID"
    echo $CMD
}

query() {
    local CMD="https://philly/api/status?"
    JOB_ID=$1
    CMD+="jobType=cust&"
    CMD+="clusterId=$CLUSTER&"
    CMD+="vcId=$VC&"
    CMD+="jobId=$JOB_ID&"
    CMD+="content=partial"

    echo $CMD
}

list() {
    if [ ! -z $1 ]; then
        STATUS=$1
    else
        STATUS=Running
    fi

    local CMD="https://philly/api/list?"
    CMD+="clusterId=$CLUSTER&"
    CMD+="jobType=cust&"
    CMD+="vcId=$VC&"
    CMD+="numFinished=1&"
    # All, Running, Passed, Failed, Completed
    CMD+="status=$STATUS&"
    CMD+="userName=$USERNAME"


    echo $CMD
}

if [ $1 = "submit" ]; then
    CMD=$(submit $2)
elif [ $1 = "query" ]; then
    CMD=$(query $2)
elif [ $1 = "abort" ]; then
    CMD=$(abort $2)
elif [ $1 = "list" ]; then
    CMD=$(list $2)
else
    echo "Action doesnot support yet."
fi
echo $CMD

curl -k --ntlm --user "$USERNAME" "$CMD"

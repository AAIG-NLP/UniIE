#!/usr/bin/env bash

# 运行本代码前请先安装quake-client客户端
# 安装方法见文档: https://yuque.antfin-inc.com/ulgf01/bbe97u/qxwvgv

code_path="git@gitlab.alibaba-inc.com:aaig_nlp/UniIE.git master"

project_name="volcano"
# 用户的工号
user_id="318442"
demand_id="1219"
timestamp=$(date +%Y%m%d%H%M%S)
job_name="UniLM-predict-${timestamp}"
train_torch_image="reg.docker.alibaba-inc.com/algorithm/quake:torch-1.8.0-cuda102-cudnn7-centos7-train-v2.0" # 2080Ti
train_tf_image="reg.docker.alibaba-inc.com/algorithm/quake:tensorflow-1.15-horovod"
# train_torch_image="reg.docker.alibaba-inc.com/algorithm/quake:torch-1.8.2-cuda111-centos7-train-v2.0" # 3090
# train_torch_image="reg.docker.alibaba-inc.com/algorithm/quake:torch-1.8.0-cuda102-cudnn7-centos7-train-v2.1-elastic"

config_path=/mnt/sijinghui.sjh/deepIE/config/quake_config_unilm_base.ini # 测试实验
entry_file=run_predict.py

quakecmd train \
--emp_id ${user_id} \
--demand_id ${demand_id} \
--job_name "${job_name}" \
--project_name "${project_name}" \
--code_url "${code_path}" \
--training_image_path "${train_torch_image}" \
--entry_file ${entry_file} \
--train_node_num "1" \
--training_start_params " --config_file ${config_path}" \
--gpu_type "2080Ti" \
--training_inputs '/mnt/sijinghui.sjh:datastore:train-nas-part1:/sijinghui.sjh,/mnt/albert.xht:datastore:train-nas-part1:/albert.xht' \


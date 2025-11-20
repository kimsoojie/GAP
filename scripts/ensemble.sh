#!/bin/bash

# Ensemble inference on NW-UCLA dataset
python utils/ensemble.py \
--config ucla \
--num-split 5 \
--dataset NW-UCLA \
--joint-dir work_dir/ucla/joint \
--bone-dir work_dir/ucla/bone \
--joint-motion-dir work_dir/ucla/joint_vel  \
--bone-motion-dir work_dir/ucla/bone_vel

# Ensemble inference on NTU-120 Cross-Set
# python utils/ensemble.py \
# --config ntu \
# --num-split 100 \
# --dataset ntu120/xset \
# --joint-dir work_dir/ntu120/cset/joint \
# --bone-dir work_dir/ntu120/cset/bone \
# --joint-motion-dir work_dir/ntu120/cset/joint_vel  \
# --bone-motion-dir work_dir/ntu120/cset/bone_vel

# Ensemble inference on NTU-120 Cross-Subject
# python utils/ensemble.py \
# --config ntu \
# --num-split 100 \
# --dataset ntu120/xsub \
# --joint-dir work_dir/ntu120/csub/joint \
# --bone-dir work_dir/ntu120/csub/bone \
# --joint-motion-dir work_dir/ntu120/csub/joint_vel  \
# --bone-motion-dir work_dir/ntu120/csub/bone_vel

# Ensemble inference on NTU-60 Cross-Subject
# python utils/ensemble.py \
# --config ntu \
# --num-split 100 \
# --dataset ntu/xsub \
# --joint-dir work_dir/ntu60/csub/joint \
# --bone-dir work_dir/ntu60/csub/bone \
# --joint-motion-dir work_dir/ntu60/csub/joint_vel  \
# --bone-motion-dir work_dir/ntu60/csub/bone_vel

# Ensemble inference on NTU-60 Cross-View
# python utils/ensemble.py \
# --config ntu \
# --num-split 100 \
# --dataset ntu/xview \
# --joint-dir work_dir/ntu60/cview/joint \
# --bone-dir work_dir/ntu60/cview/bone \
# --joint-motion-dir work_dir/ntu60/cview/joint_vel  \
# --bone-motion-dir work_dir/ntu60/cview/bone_vel
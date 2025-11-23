#!/bin/bash

modality='joint' # joint / bone / joint_vel / bone_vel

# Test on NTU-120 Cross-Set
# python main_multipart_ntu.py --config config/nturgbd120-cross-set/lst_${modality}.yaml --work-dir work_dir/ntu120/cset/${modality} --phase test --save-score True --weights work_dir/ntu120/cset/${modality}/ --device 0

# Test on NTU-120 Cross-Subject
# python main_multipart_ntu.py --config config/nturgbd120-cross-subject/lst_${modality}.yaml --work-dir work_dir/ntu120/csub/${modality} --phase test --save-score True --weights work_dir/ntu120/csub/${modality}/ --device 0

# Test on NTU-60 Cross-Subject
# python main_multipart_ntu.py --config config/nturgbd-cross-subject/lst_${modality}.yaml --work-dir work_dir/ntu60/csub/${modality} --phase test --save-score True --weights work_dir/ntu60/csub/${modality}/ --device 0

# Test on NTU-60 Cross-View
# python main_multipart_ntu.py --config config/nturgbd-cross-view/lst_${modality}.yaml --work-dir work_dir/ntu60/cview/${modality} --phase test --save-score True --weights work_dir/ntu60/cview/${modality}/ --device 0

# Test on NUCLA
python main_multipart_ucla.py --config config/ucla/lst_${modality}.yaml --work-dir work_dir/ucla/lst_${modality} --phase test --save-score True --weights work_dir/ucla/lst_${modality}/ --device 0
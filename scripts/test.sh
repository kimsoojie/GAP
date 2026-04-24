#!/bin/bash

modality='joint' # joint / bone / joint_vel / bone_vel

# Test on NTU-120 Cross-Set
# python main_multipart_ntu.py --config config/nturgbd120-cross-set/lst_${modality}.yaml --work-dir work_dir/ntu120/cset/${modality} --phase test --save-score True --weights work_dir/ntu120/cset/${modality}/ --device 0 1 2 3

# Test on NTU-120 Cross-Subject
# python main_multipart_ntu.py --config config/nturgbd120-cross-subject/lst_${modality}.yaml --work-dir work_dir/ntu120/csub/${modality} --phase test --save-score True --weights work_dir/ntu120/csub/${modality}/ --device 0 1 2 3

# Test on PKU-MMD-V1 Cross-Subject
#python main_multipart_pku.py --config config/pkuv1-cross-subject/lst_${modality}.yaml --work-dir work_dir/pkuv1/csub/${modality} --phase test --save-score True --weights work_dir/pkuv1/csub/${modality}/ --device 0 1 2 3

# Test on PKU-MMD-V2 Cross-Subject
python main_multipart_pku.py --config config/pkuv2-cross-subject/lst_${modality}.yaml --work-dir work_dir/pkuv2/csub/${modality} --phase test --save-score True --weights work_dir/pkuv2/csub/${modality}/ --device 0 1 2 3

# Test on PKU-MMD-V1 Cross-View
#python main_multipart_pku.py --config config/pkuv1-cross-view/lst_${modality}.yaml --work-dir work_dir/pkuv1/cview/${modality} --phase test --save-score True --weights work_dir/pkuv1/cview/${modality}/ --device 0 1 2 3

# Test on PKU-MMD-V2 Cross-View
# python main_multipart_pku.py --config config/pkuv2-cross-view/lst_${modality}.yaml --work-dir work_dir/pkuv2/cview/${modality} --phase test --save-score True --weights work_dir/pkuv2/cview/${modality}/ --device 0 1 2 3

# Test on NTU-60 Cross-Subject
# python main_multipart_ntu.py --config config/nturgbd-cross-subject/lst_${modality}.yaml --work-dir work_dir/ntu60/csub/${modality} --phase test --save-score True --weights work_dir/ntu60/csub/${modality}/ --device 0 1 2 3

# Test on NTU-60 Cross-View
# python main_multipart_ntu.py --config config/nturgbd-cross-view/lst_${modality}.yaml --work-dir work_dir/ntu60/cview/${modality} --phase test --save-score True --weights work_dir/ntu60/cview/${modality}/ --device 0 1 2 3

# Test on NUCLA
# python main_multipart_ucla.py --config config/ucla/lst_${modality}.yaml --work-dir work_dir/ucla/${modality} --phase test --save-score True --weights work_dir/ucla/${modality}/ --device 0 1 2 3

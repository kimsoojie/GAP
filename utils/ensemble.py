import argparse
import pickle
import os
import yaml
import numpy as np
from tqdm import tqdm
from utils import one_shot_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config',
                        required=True,
                        choices={'ucla', 'ntu'})
    parser.add_argument('--num-split',
                        required=True,
                        default=5,
                        type=int)
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)

    arg = parser.parse_args()

    dataset = arg.dataset
    
    if 'ntu' in arg.dataset:
     ## ntu
        arg.alpha = [0.4, 0.6, 0.2, 0.2]
    elif 'UCLA' in arg.dataset:
         # ucla
        arg.alpha = [0.4, 0.4, 0.3, 0.2]    
    
    if 'UCLA' in arg.dataset:
        label = []
        with open('../data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('../data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('../data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('../data/' + 'ntu60/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('../data/' + 'ntu60/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError

    with open(os.path.join(arg.joint_dir, 'test_score.pkl'), 'rb') as r1_file:
        r1_dict = pickle.load(r1_file)
        r1 = list(r1_dict['score'].items())
        r1_emb = list(enumerate(r1_dict['embedding']))
    with open(os.path.join(arg.bone_dir, 'test_score.pkl'), 'rb') as r2_file:
        r2_dict = pickle.load(r2_file)
        r2 = list(r2_dict['score'].items())
        r2_emb = list(enumerate(r2_dict['embedding']))

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, 'test_score.pkl'), 'rb') as r3_file:
            r3_dict = pickle.load(r3_file)
            r3 = list(r3_dict['score'].items())
            r3_emb = list(enumerate(r3_dict['embedding']))
    
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, 'test_score.pkl'), 'rb') as r4_file:
            r4_dict = pickle.load(r4_file)
            r4 = list(r4_dict['score'].items())
            r4_emb = list(enumerate(r4_dict['embedding']))

    right_num = total_num = right_num_5 = 0

    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            _, r44 = r4[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    elif arg.joint_motion_dir is not None and arg.bone_motion_dir is None:
        arg.alpha = [0.6, 0.6, 0.4]
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    elif arg.joint_motion_dir is None and arg.bone_motion_dir is not None:
        arg.alpha = [0.6, 0.6, 0.4]
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r44 = r4[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r44 * arg.alpha[2]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    else:
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

    # One-shot evaluation ensemble
    print('\n=== One-Shot Evaluation ===')
    
    # Prepare embeddings for ensemble
    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        ensemble_embeddings = []
        for i in range(len(label)):
            _, emb1 = r1_emb[i]
            _, emb2 = r2_emb[i]
            _, emb3 = r3_emb[i]
            _, emb4 = r4_emb[i]
            # Weighted ensemble of embeddings
            emb_ensemble = emb1 * arg.alpha[0] + emb2 * arg.alpha[1] + emb3 * arg.alpha[2] + emb4 * arg.alpha[3]
            ensemble_embeddings.append(emb_ensemble)
        ensemble_embeddings = np.array(ensemble_embeddings)
    elif arg.joint_motion_dir is not None and arg.bone_motion_dir is None:
        ensemble_embeddings = []
        for i in range(len(label)):
            _, emb1 = r1_emb[i]
            _, emb2 = r2_emb[i]
            _, emb3 = r3_emb[i]
            emb_ensemble = emb1 * arg.alpha[0] + emb2 * arg.alpha[1] + emb3 * arg.alpha[2]
            ensemble_embeddings.append(emb_ensemble)
        ensemble_embeddings = np.array(ensemble_embeddings)
    elif arg.joint_motion_dir is None and arg.bone_motion_dir is not None:
        ensemble_embeddings = []
        for i in range(len(label)):
            _, emb1 = r1_emb[i]
            _, emb2 = r2_emb[i]
            _, emb4 = r4_emb[i]
            emb_ensemble = emb1 * arg.alpha[0] + emb2 * arg.alpha[1] + emb4 * arg.alpha[2]
            ensemble_embeddings.append(emb_ensemble)
        ensemble_embeddings = np.array(ensemble_embeddings)
    else:
        ensemble_embeddings = []
        for i in range(len(label)):
            _, emb1 = r1_emb[i]
            _, emb2 = r2_emb[i]
            emb_ensemble = emb1 * arg.alpha[0] + emb2 * arg.alpha[1]
            ensemble_embeddings.append(emb_ensemble)
        ensemble_embeddings = np.array(ensemble_embeddings)

    oneshot_results = one_shot_evaluation(config=arg.config, unseen_split=arg.num_split, llm_embeddings=ensemble_embeddings, labels=label)
    print(f"One-shot Accuracy - Total: {oneshot_results['total']:.4f}, Seen: {oneshot_results['seen']:.4f}, Unseen: {oneshot_results['unseen']:.4f}")


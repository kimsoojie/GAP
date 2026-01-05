import argparse
import random
import sys
import traceback
import numpy as np
import torch

def get_label_split_oneshot(config, num_split):
    if 'NTU' in config:
        unique_labels = list(range(120))
        base_unseen_labels = list(range(0, 120, 6)) 
        
        if num_split == 20: num_seen=20
        elif num_split == 40: num_seen=40
        elif num_split == 60: num_seen=60
        elif num_split == 80: num_seen=80
        elif num_split == 100: num_seen=100
        
        remaining_labels = [l for l in unique_labels if l not in base_unseen_labels]
        if len(remaining_labels) >= num_seen:
            indices = np.linspace(0, len(remaining_labels)-1, num_seen, dtype=int)
            seen_labels = [remaining_labels[i] for i in indices]
        else:
            seen_labels = remaining_labels
        unseen_labels = base_unseen_labels  # Changed from set to list
        print(f"Unseen labels: {sorted(unseen_labels)}")
        print(f"Seen labels: {sorted(seen_labels)}")
        
    elif 'NW-UCLA' in config:
        unique_labels = list(range(10))
        base_unseen_labels = list(range(0, 10, 2))  # [0, 2, 4, 6, 8]

        if num_split == 5: 
            num_seen = 5 # [1, 3, 5, 7, 9]
        elif num_split == 3: 
            num_seen = 3 # [1, 5, 9]
        else:
            raise ValueError(f"Unsupported split: {num_split}")

        remaining_labels = [l for l in unique_labels if l not in base_unseen_labels]
        if len(remaining_labels) >= num_seen:
            indices = np.linspace(0, len(remaining_labels)-1, num_seen, dtype=int)
            seen_labels = [remaining_labels[i] for i in indices]
        else:
            seen_labels = remaining_labels
        unseen_labels = base_unseen_labels  # Changed from set to list

        print(f"Unseen labels: {sorted(unseen_labels)}")
        print(f"Seen labels: {sorted(seen_labels)}")
    
    return seen_labels, unseen_labels

def get_label_split_zsl(config, num_split):
    if 'NTU120' in config:
        all_labels = list(range(120))
        if num_split == 10: 
            unseen_labels =  [4, 13, 37, 43, 49, 65, 88, 95, 99, 106]
        elif num_split == 24: 
            unseen_labels =   [5, 9, 11, 16, 18, 20, 22, 29, 35, 39, 45, 49, 59, 68, 70, 81, 84, 87, 93, 94, 104, 113, 114, 119]
        elif num_split == 40:
            unseen_labels =  [11, 12, 18, 22, 23, 26, 28, 34, 37, 38, 42, 44, 46, 47, 48, 57, 59, 64, 66, 70, 73, 74, 75, 83, 86, 90, 92, 93, 95, 96, 102, 104, 107, 108, 110, 112, 115, 116, 118, 119]
        elif num_split == 60:
            unseen_labels =  [0, 1, 4, 6, 7, 8, 9, 17, 18, 21, 23, 25, 26, 28, 30, 32, 33, 34, 37, 38, 39, 40, 41, 42, 44, 45, 50, 51, 52, 53, 56, 61, 62, 65, 67, 68, 69, 70, 74, 77, 78, 81, 83, 87, 89, 90, 91, 92, 94, 95, 96, 97, 100, 101, 109, 111, 114, 115, 116, 118]
        
        seen_labels = [l for l in all_labels if l not in unseen_labels]
    
    elif 'NTU60' in config:  
        all_labels = list(range(60))
        if num_split == 5: 
            unseen_labels = [10, 11, 19, 26, 56]
        elif num_split == 12: 
            unseen_labels = [3, 5, 9, 12, 15, 40, 42, 47, 51, 56, 58, 59]
        elif num_split == 20:
            unseen_labels = [0, 12, 13, 14, 15, 16, 17, 22, 23, 26, 29, 30, 31, 35, 36, 42, 43, 48, 56, 57]
        elif num_split == 30:
            unseen_labels =  [0, 1, 2, 6, 7, 8, 10, 12, 13, 15, 16, 18, 20, 21, 25, 26, 27, 31, 32, 33, 39, 42, 45, 47, 48, 51, 52, 55, 58, 59]
        
        seen_labels = [l for l in all_labels if l not in unseen_labels]
        
    elif 'NW-UCLA' in config:
        
        if num_split == 3:
            seen_labels = [1,5,9]
            unseen_labels = [0,2,4,6,8]
        elif num_split == 5:
            seen_labels = [1,3,5,7,9]
            unseen_labels = [0,2,4,6,8]
    
    return seen_labels, unseen_labels

def one_shot_evaluation(config='NW-UCLA', unseen_split=5, llm_embeddings=None, labels=None, num_trials=500):
        """
        One-shot learning evaluation: randomly select one reference sample per label
        and predict labels based on cosine similarity.
        """
        config =config
        unseen_split = unseen_split
        seen_labels, unseen_labels = get_label_split_oneshot(config, unseen_split)
        
        # Ensure numpy arrays for indexing
        if isinstance(llm_embeddings, torch.Tensor):
            llm_embeddings = llm_embeddings.float().cpu().numpy()
        elif not isinstance(llm_embeddings, np.ndarray):
            llm_embeddings = np.array(llm_embeddings)
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        elif not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        def evaluate_subset(subset_labels, embeddings, labels):
            """Helper function to evaluate on a specific subset of labels"""
            # Ensure numpy arrays for indexing
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            # Filter samples belonging to subset_labels
            subset_mask = np.isin(labels, subset_labels)
            subset_embeddings = embeddings[subset_mask]
            subset_sample_labels = labels[subset_mask]
            
            if len(subset_sample_labels) == 0:
                return 0.0
            
            # Select one reference sample per label in subset
            unique_labels = np.unique(subset_sample_labels)
            reference_indices = []
            reference_labels = []
            
            for label in unique_labels:
                label_indices = np.where(subset_sample_labels == label)[0]
                ref_idx = np.random.choice(label_indices)
                reference_indices.append(ref_idx)
                reference_labels.append(label)
            
            reference_embeddings = subset_embeddings[reference_indices]
            reference_labels = np.array(reference_labels)
            
            # Create test set (exclude reference samples)
            all_indices = np.arange(len(subset_sample_labels))
            test_mask = ~np.isin(all_indices, reference_indices)
            test_embeddings = subset_embeddings[test_mask]
            test_labels = subset_sample_labels[test_mask]
            
            if len(test_labels) == 0:
                return 0.0
            
            # Normalize embeddings for cosine similarity
            reference_embeddings_norm = reference_embeddings / (np.linalg.norm(reference_embeddings, axis=1, keepdims=True) + 1e-8)
            test_embeddings_norm = test_embeddings / (np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-8)
            
            # Compute cosine similarity
            similarity = np.dot(test_embeddings_norm, reference_embeddings_norm.T)
            
            # Predict labels
            predicted_indices = np.argmax(similarity, axis=1)
            predicted_labels = reference_labels[predicted_indices]
            
            # Calculate accuracy
            correct = np.sum(predicted_labels == test_labels)
            accuracy = correct / len(test_labels)
            
            return accuracy
        
        # Run multiple trials and average the results
        seen_accuracies = []
        unseen_accuracies = []
        total_accuracies = []
        
        for trial in range(num_trials):
            # Evaluate on seen labels only
            accuracy_seen = evaluate_subset(seen_labels, llm_embeddings, labels)
            seen_accuracies.append(accuracy_seen)
            
            # Evaluate on unseen labels only
            accuracy_unseen = evaluate_subset(unseen_labels, llm_embeddings, labels)
            unseen_accuracies.append(accuracy_unseen)
            
            # Evaluate on all labels (seen + unseen)
            all_labels = np.concatenate([np.array(seen_labels).flatten(), np.array(unseen_labels).flatten()])
            accuracy_total = evaluate_subset(all_labels, llm_embeddings, labels)
            total_accuracies.append(accuracy_total)
        
        # Calculate average accuracies
        avg_accuracy_seen = np.mean(seen_accuracies)
        avg_accuracy_unseen = np.mean(unseen_accuracies)
        avg_accuracy_total = np.mean(total_accuracies)
        
        return {
            'total': avg_accuracy_total,
            'seen': avg_accuracy_seen,
            'unseen': avg_accuracy_unseen
        }
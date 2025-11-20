import argparse
import random
import sys
import traceback
import numpy as np
import torch

def get_label_split_oneshot(config, num_split):
    if 'ntu' in config:
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
        
    elif 'ucla' in config:
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
    if 'ntu' in config:
        # seen/unseen split
        # split 115/5: unseen [0, 24, 48, 72, 96]
        # split 110/10: unseen [0, 12, 24, 36, 48, 60, 72, 84, 96, 108]
        # split 96/24: unseen [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
        # split 80/40: unseen [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99, 102, 105, 108, 111, 114, 117]
        # split 60/60: unseen [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118]

        unique_labels = list(range(120))
        if num_split == 5: num_unseen=5
        elif num_split == 10: num_unseen=10
        elif num_split == 24: num_unseen=24
        elif num_split == 40: num_unseen=40
        elif num_split == 60: num_unseen=60
        
        step = max(1, len(unique_labels) // num_unseen)
        unseen_labels = unique_labels[::step][:num_unseen]
        seen_labels = [l for l in unique_labels if l not in unseen_labels]
        print(f"Unseen labels: {sorted(unseen_labels)}")
        print(f"Seen labels: {sorted(seen_labels)}")
        
    elif 'ucla' in config:
        unique_labels = list(range(10))
        num_unseen = 5
        step = max(1, len(unique_labels) // num_unseen)
        unseen_labels = unique_labels[::step][:num_unseen]
        seen_labels = [l for l in unique_labels if l not in unseen_labels]
        print(f"Unseen labels: {sorted(unseen_labels)}") # [0, 2, 4, 6, 8]
        print(f"Seen labels: {sorted(seen_labels)}") # [1, 3, 5, 7, 9]
    
    return seen_labels, unseen_labels

def one_shot_evaluation(self, config='ucla', unseen_split=5, llm_embeddings=None, labels=None, num_trials=500):
        """
        One-shot learning evaluation: randomly select one reference sample per label
        and predict labels based on cosine similarity.
        
        Args:
            args: arguments containing config and unseen_split
            llm_embeddings: torch.Tensor or np.ndarray of shape (N, D)
            labels: torch.Tensor or np.ndarray of shape (N,)
            num_trials: number of trials to average over
        
        Returns:
            dict: accuracies for 'total', 'seen', 'unseen'
        """
        config =config
        unseen_split = unseen_split
        seen_labels, unseen_labels = self.get_label_split_oneshot(config, unseen_split)
        
        if isinstance(llm_embeddings, torch.Tensor):
            llm_embeddings = llm_embeddings.float().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        def evaluate_subset(subset_labels, embeddings, labels):
            """Helper function to evaluate on a specific subset of labels"""
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
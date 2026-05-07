import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def save_gmm_plot(scores, preds, means, save_path):
    """
    绘制 GMM 分离效果直方图
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图，根据预测结果着色
    clean_scores = scores[preds == 0]
    poison_scores = scores[preds == 1]
    
    plt.hist([clean_scores, poison_scores], bins=30, stacked=True, 
             color=['green', 'red'], label=['Predicted Clean', 'Predicted Poison'], alpha=0.7, edgecolor='black')
    
    # 标记 GMM 均值
    for i, mean in enumerate(means):
        plt.axvline(mean, color='blue', linestyle='--', linewidth=2, label=f'GMM Mean {i+1}: {mean:.2f}')
    
    plt.title('GMM Separation on Suspicious Set (P1)', fontsize=14)
    plt.xlabel('Dominant Column Sum (Score)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"   -> [Saved] GMM separation plot to {save_path}")

def detect_poison_samples(npz_path):
    # 确定保存目录 (与 npz 文件同级)
    save_dir = os.path.dirname(npz_path)
    
    print(f"===> Loading transition matrices from: {npz_path}")
    if not os.path.exists(npz_path):
        print(f"Error: File not found at {npz_path}")
        return

    data = np.load(npz_path)
    all_T = data['all_T']                 
    all_is_poison = data['all_is_poison'] 
    print(f"真实中毒样本数: {np.sum(all_is_poison)} / {len(all_is_poison)}")
    
    if 'norm_total_T' in data:
        norm_total_T = data['norm_total_T']
    else:
        total_sum_T = np.sum(all_T, axis=0)
        row_sums = np.sum(total_sum_T, axis=1, keepdims=True)
        norm_total_T = total_sum_T / (row_sums + 1e-12)
    
    num_samples = all_T.shape[0]
    num_classes = all_T.shape[1]

    # --- Step 1: 确定目标类 ---
    mask_off_diag = 1 - np.eye(num_classes)
    off_diag_sums = np.sum(norm_total_T * mask_off_diag, axis=0)
    predicted_target_class = np.argmax(off_diag_sums)
    
    print(f"\n[Step 1] Target Class Identification: Predicted Class {predicted_target_class}")

    # --- Step 2: 筛选 P1 集合 ---
    suspicious_indices = []
    suspicious_scores = []
    
    for i in range(num_samples):
        # col_sums = np.sum(all_T[i], axis=0)
        # max_col_idx = np.argmax(col_sums)
        
        # if max_col_idx == predicted_target_class:
        #     suspicious_indices.append(i)
        #     suspicious_scores.append(col_sums[predicted_target_class])
        
        # 计算非对角线元素和
        off_diag_sums = np.sum(all_T[i] * mask_off_diag, axis=0)
        max_col_idx = np.argmax(off_diag_sums)
    
        # 判断是否属于目标类
        if max_col_idx == predicted_target_class:
            suspicious_indices.append(i)
            suspicious_scores.append(off_diag_sums[predicted_target_class])
            
    suspicious_indices = np.array(suspicious_indices)
    suspicious_scores_arr = np.array(suspicious_scores) # 保持一维用于画图
    suspicious_scores = suspicious_scores_arr.reshape(-1, 1) # GMM 需要二维
    
    print(f"[Step 2] P1 Set Size: {len(suspicious_indices)}")
    
    if len(suspicious_scores) < 5:
        print("Error: Too few samples in P1.")
        return

    # --- Step 3: GMM 分离与硬阈值判决 ---
    # 1. 使用 GMM 寻找两个簇的中心 (自动定位)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(suspicious_scores)
    
    means = gmm.means_.flatten()
    clean_mean = np.min(means)
    poison_mean = np.max(means)
    
    # 2. 计算硬阈值 (Hard Threshold)
    # 取两个均值的中间点，或者为了更激进地召回，可以设为 (clean_mean + 2*poison_mean)/3 等，但通常中点就够了
    #hard_threshold = (clean_mean + poison_mean) / 2
    hard_threshold = (clean_mean + 2*poison_mean)/3
    
    print(f"\n[Step 3] Separation Logic Adjustment:")
    print(f"   -> GMM Clean Mean: {clean_mean:.4f}")
    print(f"   -> GMM Poison Mean: {poison_mean:.4f}")
    print(f"   -> Calculated Hard Threshold: {hard_threshold:.4f}")
    print(f"   -> Strategy: Rejecting soft GMM probability. Using Threshold > {hard_threshold:.4f}")

    # 3. 执行硬阈值判决
    p1_preds = (suspicious_scores_arr > hard_threshold).astype(int)
    
    
    
    # 全局预测更新
    final_preds = np.zeros(num_samples)
    final_scores = np.zeros(num_samples)
    
    # 标记预测为中毒的样本
    final_preds[suspicious_indices[p1_preds == 1]] = 1
    
    final_scores[suspicious_indices] = suspicious_scores_arr 

    # --- 保存可视化图表 ---
    plot_path = os.path.join(save_dir, 'gmm_separation.png')
    save_gmm_plot(suspicious_scores_arr, p1_preds, means, plot_path)

    # --- Step 4: 评估与保存文本报告 ---
    
    # 计算指标
    report = classification_report(all_is_poison, final_preds, target_names=['Clean', 'Poisoned'], digits=4)
    auc = roc_auc_score(all_is_poison, final_scores)
    cm = confusion_matrix(all_is_poison, final_preds)
    
    pred_poisons_count = final_preds.sum()
    tp = np.sum((final_preds == 1) & (all_is_poison == 1))
    fp = np.sum((final_preds == 1) & (all_is_poison == 0))
    fn = np.sum((final_preds == 0) & (all_is_poison == 1))
    
    # 构建报告文本
    summary_text = [
        "==================================================",
        f"BLTM Detection Report",
        f"Target Class: {predicted_target_class}",
        "==================================================",
        f"\n[GMM Statistics on P1 Set]",
        f"P1 Set Size: {len(suspicious_indices)}",
        f"GMM Means: {means} (Larger mean is Poison cluster)",
        f"\n[Detection Performance]",
        f"AUC Score: {auc:.4f}",
        f"Predicted Poisons: {int(pred_poisons_count)}",
        f"True Positives (TP): {int(tp)}",
        f"False Positives (FP): {int(fp)}",
        f"False Negatives (FN): {int(fn)}",
        f"\n[Classification Report]\n{report}",
        f"\n[Confusion Matrix]\n{cm}",
        "=================================================="
    ]
    
    # 打印到控制台
    print("\n".join(summary_text))
    
    # 保存到文件
    report_path = os.path.join(save_dir, 'detection_summary.txt')
    with open(report_path, "w") as f:
        f.write("\n".join(summary_text))
    print(f"\n   -> [Saved] Detection summary report to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default=None)
    parser.add_argument('--path', type=str, default=None)
    args = parser.parse_args()
    
    if args.path:
        target_path = args.path
    elif args.result_file:
        target_path = os.path.join('record', args.result_file, 'detection', 'bltm_detect', 'all_T_matrix.npz')
    else:
        target_path = 'record/1229_badnet_2_0.1_cifar10_pre18/detection/bltm_detect/all_T_matrix.npz'
        print("No finfing path provided, using default path:", target_path)
    
    detect_poison_samples(target_path)
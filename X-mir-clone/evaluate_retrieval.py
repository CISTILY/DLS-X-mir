import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

def evaluate_retrieval(test_file, retrieval_file, qrel_file, k=10):
    """
    Đánh giá hệ thống truy vấn dựa trên các file đầu vào, bao gồm cả micro-F1 và macro-F1.
    
    Args:
        test_file (str): Đường dẫn đến file test.txt chứa query và label.
        retrieval_file (str): Đường dẫn đến file retrievalResult.txt chứa kết quả truy vấn.
        qrel_file (str): Đường dẫn đến file qrel_0_1.txt chứa thông tin liên quan.
        k (int): Ngưỡng K để tính Precision@K và Recall@K.
    """

    # 1. Đọc và xử lý file test.txt
    print("Đọc file test.txt...")
    query_labels = {}
    with open(test_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[1]
            label = parts[2]
            query_labels[filename] = label
    print(f"Đã đọc {len(query_labels)} ảnh query từ test.txt.")
    
    # 2. Đọc và xử lý file retrievalResult.txt
    print("Đọc file retrievalResult.txt...")
    retrieval_data = {}
    with open(retrieval_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            query_path = parts[0]
            query_name = query_path.split('\\')[-1].split('/')[-1]
            retrieved_paths = parts[1:]
            retrieved_names = [p.split('\\')[-1].split('/')[-1] for p in retrieved_paths]
            retrieval_data[query_name] = retrieved_names
    print(f"Đã đọc kết quả truy vấn cho {len(retrieval_data)} ảnh query.")
    
    # 3. Đọc và xử lý file qrel_0_1.txt
    print("Đọc file qrel_0_1.txt...")
    qrel_df = pd.read_csv(qrel_file, sep='\t', header=None, names=['label', 'col2', 'filename', 'relevance'])
    qrel_relevance = dict(zip(qrel_df['filename'], qrel_df['label']))
    print(f"Đã đọc {len(qrel_relevance)} dòng liên quan từ qrel_0_1.txt.")
    
    # 4. Tính toán các metric
    print(f"\nBắt đầu tính toán các metric (K={k})...")
    
    total_precision_at_k = 0
    total_recall_at_k = 0
    num_queries_with_retrievals = 0
    average_precisions = []

    # Danh sách để tính macro-F1 và micro-F1
    all_y_true_at_k = []
    all_y_pred_at_k = []
    
    for query_name, retrieved_names in retrieval_data.items():
        if not retrieved_names or query_name not in query_labels:
            continue
        
        num_queries_with_retrievals += 1
        
        query_label = query_labels.get(query_name)
        relevant_images = [fname for fname, label in query_labels.items() if label == query_label]

        # Lấy top K kết quả truy vấn
        top_k_retrieved = retrieved_names[:k]

        # Xác định các kết quả liên quan trong top K
        relevant_in_top_k = 0
        y_true_k = []
        y_pred_k = []
        
        for retrieved_name in top_k_retrieved:
            # Logic kiểm tra sự liên quan
            is_relevant = False
            if query_label == 'positive' and qrel_relevance.get(retrieved_name) == 1:
                is_relevant = True
            elif query_label == 'negative' and qrel_relevance.get(retrieved_name) == 0:
                is_relevant = True
            
            y_pred_k.append(1) # Tất cả các kết quả truy xuất đều được dự đoán là relevant
            y_true_k.append(int(is_relevant))
            
            if is_relevant:
                relevant_in_top_k += 1

        all_y_true_at_k.extend(y_true_k)
        all_y_pred_at_k.extend(y_pred_k)
        
        # Tính Precision@K cho query hiện tại
        precision_at_k = relevant_in_top_k / k
        total_precision_at_k += precision_at_k

        # Tính Recall@K cho query hiện tại
        if relevant_images:
            recall_at_k = relevant_in_top_k / len(relevant_images)
            total_recall_at_k += recall_at_k

        # Tính Average Precision (AP)
        y_true_ap = []
        for retrieved_name in retrieved_names:
            is_relevant = False
            if query_label == 'positive' and qrel_relevance.get(retrieved_name) == 1:
                is_relevant = True
            elif query_label == 'negative' and qrel_relevance.get(retrieved_name) == 0:
                is_relevant = True
            y_true_ap.append(int(is_relevant))
        
        if sum(y_true_ap) > 0:
            ap = average_precision_score(y_true_ap, range(len(y_true_ap)))
            average_precisions.append(ap)
            
    # Tính trung bình các metric
    mean_precision_at_k = total_precision_at_k / num_queries_with_retrievals
    print(mean_precision_at_k)
    mean_recall_at_k = total_recall_at_k / num_queries_with_retrievals
    mean_average_precision = sum(average_precisions) / len(average_precisions) if average_precisions else 0

    # Tính micro-F1
    micro_f1 = f1_score(all_y_true_at_k, all_y_pred_at_k, average='micro')

    print("\nKết quả đánh giá:")
    print(f"Mean Precision@{k}: {mean_precision_at_k:.4f}")
    print(f"Mean Recall@{k}: {mean_recall_at_k:.4f}")
    print(f"Mean Average Precision (mAP): {mean_average_precision:.4f}")
    print(f"Micro-F1@{k}: {micro_f1:.4f}")

# Gọi hàm với các file của bạn
evaluate_retrieval('test.txt', 'retrievalResult.txt', 'qrel_0_1.txt', k=3)
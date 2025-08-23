import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
main_path = 'D:\DLS-X-mir\X-mir-clone\saliency'       # saliency maps
retrieval_file = 'D:/DLS-X-mir/SearchAndIndexingJavaCode/retrievalResult.txt'

# Dicts to store results
sal_map = {}
ret_map = {}
key_dict = {}
ins_del_q_dict = {}
class_labels = {}

# Dummy metric class (replace with real one)
class InsertDeleteMetric:
    def forward(self, q_tensor, retrieved_tensors, saliency_maps):
        insertion = [float(np.random.rand()) for _ in retrieved_tensors]
        deletion = [float(np.random.rand()) for _ in retrieved_tensors]
        return insertion, deletion, insertion, deletion

get_insert_dele = InsertDeleteMetric()

# Storage collectors
class AvgCollector:
    def __init__(self):
        self.data = {}

    def store(self, cls, value):
        if cls not in self.data:
            self.data[cls] = []
        self.data[cls].append(value)

ins_avg_c = AvgCollector()
del_avg_c = AvgCollector()

# ----------------------------
# Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def prep_image_(img_path):
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

# ----------------------------
# Load retrieval results
# ----------------------------
retrieval_results = {}
with open(retrieval_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        query_path = parts[0]       # full path to query
        retrieved_paths = parts[1:] # full paths to retrieved
        retrieval_results[query_path] = retrieved_paths

# ----------------------------
# Main loop
# ----------------------------
for query_path, retrieved_list in retrieval_results.items():
    query_name = os.path.basename(query_path)
    print("Processing query:", query_name)

    if not os.path.exists(query_path):
        print(f"Query image not found: {query_path}")
        continue

    query_tensor = prep_image_(query_path)
    q_key = os.path.splitext(query_name)[0]

    sal_map[q_key] = []
    ret_map[q_key] = []
    key_dict[q_key] = []

    for r_path in retrieved_list:
        r_name = os.path.basename(r_path)
        r_key = os.path.splitext(r_name)[0]

        # Load saliency map (assumed structure: saliency/<query_name>/<retrieved_name>.npy)
        sal_file = os.path.join(main_path, query_name, r_name + ".npy")
        if not os.path.exists(sal_file):
            print(f"Missing saliency: {sal_file}")
            continue
        sal_map[q_key].append(np.load(sal_file))

        # Load retrieved image (full path already given in file)
        if not os.path.exists(r_path):
            print(f"Missing retrieved image: {r_path}")
            continue
        ret_map[q_key].append(prep_image_(r_path))

        key_dict[q_key].append(r_name)

    if len(ret_map[q_key]) == 0:
        print(f"No retrieved images processed for {query_name}")
        continue

    # Compute local deletion and insertion metric
    insertion, deletion, _, _ = get_insert_dele.forward(
        query_tensor, ret_map[q_key], sal_map[q_key]
    )

    avg_insert = sum(insertion) / len(insertion)
    avg_del = sum(deletion) / len(deletion)
    print("Avg insertion:", avg_insert, " Avg deletion:", avg_del)

    ins_del_q_dict[q_key] = [insertion, deletion]

    # Assign dummy class label (replace if you have label dict)
    cls_label = class_labels.get(query_name, "unknown")
    ins_avg_c.store(cls_label, avg_insert)
    del_avg_c.store(cls_label, avg_del)

    del sal_map[q_key]

print("Finished processing all queries.")

# import os
# import math
# import json
# import torch
# import numpy as np
# from PIL import Image
# import torch.nn as nn
# from model import DenseNet121
# from torchvision import transforms
# from evaluation import CausalMetric, gkern


# # NOTE: Edit the dataset_type and path to model_weights here
# dataset_type = 'covid'
# model_weights = 'D:/DLS-X-mir/X-mir-clone/checkpoints/covid_densenet121_seed_0_epoch_20_ckpt.pth'


# class InsDel():
#     def __init__(self,
#                  model):
#         self.model = model
#         net_in_size = 224
#         klen = 51
#         ksig = math.sqrt(50)
#         kern = gkern(klen, ksig)
#         def blur(x): return nn.functional.conv2d(
#             x, kern.cuda(), padding=klen//2)
#         self.insertion = CausalMetric(
#             self.model, 'ins', net_in_size, substrate_fn=blur)
#         self.deletion = CausalMetric(
#             self.model, 'del', net_in_size, substrate_fn=torch.zeros_like)

#     def evaluate(self, new_sal, ret_image):
#         """
#         This function evaluates an image and its saliency map for deletion 
#         and insertion.

#         Attributes:
#             new_sal (numpy.array): The saliency map obtained at iteration k. 
#             ret_image (torch.tensor.cuda): Input image that needs to be explained.

#         Returns:
#         score_del (float): The deletion score between 0-1 range.
#         score_ins (float): The insertion score between 0-1 range.
#         """
#         new_sal = torch.from_numpy(new_sal).float()
#         score_del = 0
#         zero_cnt_del = 0
#         score_del, zero_cnt_ins = self.deletion.single_run(self.q_image.cuda(),
#                                                            ret_image.cuda(), new_sal, verbose=0)
#         score_ins, zero_cnt_del = self.insertion.single_run(self.q_image.cuda(),
#                                                             ret_image.cuda(),  new_sal, verbose=0)
#         return score_del, score_ins, zero_cnt_ins, zero_cnt_del

#     def load_query(self, query_image):
#         self.q_image = query_image

#     def forward(self, q_image, ret_dict, sal_dict):
#         ins_avg = []
#         del_avg = []
#         z_ins_list = []
#         z_del_list = []

#         self.load_query(q_image)
#         for i, sal_m in enumerate(sal_dict):
#             ret_image = ret_dict[i]
#             dele, ins, z_ins, z_del = self.evaluate(sal_m, ret_image)
#             z_ins_list.append(z_ins)
#             z_del_list.append(z_del)
#             ins_avg.append(ins)
#             del_avg.append(dele)
#         return ins_avg, del_avg, z_ins_list, z_del_list


# # Class for average counter
# class AverageCounter():
#     def __init__(self):
#         self.average = 0
#         self.running_avg = 0
#         self.fina_dict = {}

#     def store(self, q_label, metric,  k=20):
#         def check_if_in_dict(value1):
#             if value1 in self.fina_dict.keys():
#                 return True
#             else:
#                 return False
#         if check_if_in_dict(q_label):
#             self.fina_dict[q_label].append(metric)
#         else:
#             self.fina_dict[q_label] = [metric]

#     def read_Average(self):
#         avgdict = {}
#         for k, v in self.fina_dict.items():
#             avgdict[k] = sum(v) / float(len(v))
#         return avgdict


# def prep_image_(file_n):
#     query_image = Image.open(os.path.join(
#         query_img_path, file_n)).convert('RGB')
#     query_image_tensor = transform(query_image).unsqueeze_(0).cuda()
#     return query_image_tensor


# model = DenseNet121()
# model.load_state_dict(torch.load(model_weights), strict=False)
# model = model.eval()
# model = model.cuda()

# # Logging counter
# ins_avg_c = AverageCounter()
# del_avg_c = AverageCounter()


# sal_map = {}
# ret_map = {}
# query_image_list = []
# class_labels = {}

# if dataset_type == 'covid':
#     # NOTE: Edit the path to saliency maps and images here
#     main_path = 'D:/DLS-X-mir/X-mir-clone/saliency'
#     query_img_path = 'D:/DLS-X-mir/covidx-cxr2/versions/9/test'
#     valid_class = ['positive', 'negative']
#     # Unravel labels here
#     with open('test_COVIDx4.txt', 'r') as f:
#         for line in f.readlines():
#             label = line.split()[2]
#             if label not in valid_class:
#                 label = 'normal'
#             q_na = line.split()[1]
#             class_labels[q_na] = label

# if dataset_type == 'isic':
#     # NOTE: Edit the path to saliency maps and images here
#     main_path = '/data/brian.hu/isic_saliency/sbsm'
#     query_img_path = '/data/brian.hu/isic/ISIC-2017_Test_v2_Data'
#     import pandas as pd
#     valid_class = ['melanoma', 'seborrheic_keratosis']
#     df = pd.read_csv('./ISIC-2017_Test_v2_Part3_GroundTruth_balanced.csv')
#     for kj, im_name in enumerate(df['image_id']):
#         if (df['melanoma'][kj] + df['seborrheic_keratosis'][kj]) > 0:
#             if df['melanoma'][kj]:
#                 class_labels[im_name+'.jpg'] = 'melanoma'
#             else:
#                 class_labels[im_name+'.jpg'] = 'seborrheic_keratosis'
#         else:
#             class_labels[im_name+'.jpg'] = 'nevi'

# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# # NOTE: Edit the final output json files here
# f = open('D:/DLS-X-mir/X-mir-clone/inser_dele_wacv_test_simatt.json', 'w')
# f2 = open('D:/DLS-X-mir/X-mir-clone/key_list_wacv_test_simatt.json', 'w')
# ins_del_q_dict = {}
# key_dict = {}
# get_insert_dele = InsDel(model)
# for file_n in os.listdir(main_path):
#     print(file_n)
#     query_image_tensor = prep_image_(file_n)
#     retrieval_names = os.listdir(os.path.join(main_path, file_n))
#     for r_n in retrieval_names:
#         try:
#             key_dict[file_n.split('/')[0].split('.')[0]] = [r_n]
#             sal_map[file_n.split(
#                 '/')[0].split('.')[0]].append(np.load(os.path.join(main_path, file_n, r_n)))
#             ret_map[file_n.split('/')[0].split('.')[0]].append(prep_image_(os.path.join(query_img_path,
#                                                                                         '.'.join(r_n.split('.')[:-1]))))
#         except KeyError:
#             key_dict[file_n.split('/')[0].split('.')[0]].append(r_n)
#             sal_map[file_n.split('/')[0].split('.')[0]
#                     ] = [np.load(os.path.join(main_path, file_n, r_n))]
#             ret_map[file_n.split('/')[0].split('.')[0]] = [prep_image_(os.path.join(query_img_path,
#                                                                                     '.'.join(r_n.split('.')[:-1])))]

#     # Compute local deletion and insertion metric
#     insertion, deletion, i_in, i_del = get_insert_dele.forward(query_image_tensor, ret_map[file_n.split(
#         '/')[0].split('.')[0]], sal_map[file_n.split('/')[0].split('.')[0]])

#     avg_insert, avg_del = (sum(insertion)/len(insertion)
#                            ), (sum(deletion)/len(deletion))
#     # Aggerate calculate metric to associated class
#     print(avg_insert, avg_del)
#     try:
#         assert ins_del_q_dict[file_n.split('/')[0].split('.')[0]]
#         ins_del_q_dict[file_n.split(
#             '/')[0].split('.')[0]].append([insertion, deletion])
#     except KeyError:
#         ins_del_q_dict[file_n.split(
#             '/')[0].split('.')[0]] = [insertion, deletion]
#     ins_avg_c.store(class_labels[file_n], avg_insert)
#     del_avg_c.store(class_labels[file_n], avg_del)
#     del sal_map[file_n.split('/')[0].split('.')[0]]

# json.dump(ins_del_q_dict, f)
# json.dump(key_dict, f2)
# print(ins_avg_c.read_Average())
# print(del_avg_c.read_Average())

# Modify this code to get the query name which from the test folder and retrieved images which come from train folder


import os
from PIL import Image
import json
from torchvision import models, transforms
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
from tqdm import tqdm
from transformers import SegformerForImageClassification
from functools import partial
import shutil
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import json
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from scipy.stats import ks_2samp

class ImageModelEvaluator:
    def __init__(self, model_config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform = self._load_model_and_transform(model_config)
        self.imagenet_class_index = self._load_imagenet_class_index()
        print(f"Using device: {self.device}")

    def _load_model_and_transform(self, model_config):
        model_name = model_config['name']
        if model_name == 'ResNet50':
            model = models.resnet50(pretrained=True).to(self.device)
            transform = self._resnet_transform()
        elif model_name == 'ViT':
            # Placeholder for ViT model loading and its transform
            pass
        elif model_name == 'Swin':
            # Placeholder for Swin model loading and its transform
            pass
        # Additional model configurations can be added here
        model.eval()
        return model, transform

    def _resnet_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])

    def _load_imagenet_class_index(self):
        with open("index/imagenet_class_index.json", "r") as f:
            return json.load(f)

    def ensure_rgb(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img

    def evaluate(self, dataset_paths):
        all_predicted_probs = []
        for dataset_path in dataset_paths:
            dataset = self.load_images_from_directory(dataset_path)
            predicted_probs = []
            for img, label, _ in tqdm(dataset):
                img = self.ensure_rgb(img)
                img_tensor = self.transform(img).to(self.device)

                with torch.no_grad():
                    logits = self.model(img_tensor.unsqueeze(0))
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

                predicted_probs.append(probabilities)
            all_predicted_probs.append(predicted_probs)
        return all_predicted_probs

    def load_images_from_directory(self, root_path):
        dataset = []
        for label in os.listdir(root_path):
            label_path = os.path.join(root_path, label)
            if os.path.isdir(label_path):
                for image_file in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_file)
                    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img = Image.open(image_path)
                        dataset.append((img, label, image_file))
        return dataset

    def evaluate_and_save_metrics(self, dataset_paths, target_dir):
        original_dataset_probs = None

        for dataset_path in dataset_paths:
            dataset = self.load_images_from_directory(dataset_path)
            true_labels, predicted_probs, predicted_labels = [], [], []

            for img, label, _ in tqdm(dataset):
                img = self.ensure_rgb(img)
                img_tensor = self.transform(img).to(self.device)

                with torch.no_grad():
                    logits = self.model(img_tensor.unsqueeze(0))
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

                index_str, _ = self.imagenet_class_index.get(label, (None, None))
                if index_str is None:
                    continue

                true_label = int(index_str)
                true_labels.append(true_label)
                predicted_probs.append(probabilities)
                predicted_labels.append(np.argmax(probabilities))

            self.save_metrics(predicted_probs, true_labels, predicted_labels, dataset_path, target_dir, original_dataset_probs)

            if dataset_path == dataset_paths[0]:
                original_dataset_probs = [max(probs) for probs in predicted_probs]


    def save_metrics(self, predicted_probs, true_labels, predicted_labels, dataset_path, target_dir, original_dataset_probs):
        num_classes = 1000  # 或根据实际情况调整类别数
        true_labels_binary = label_binarize(true_labels, classes=range(num_classes))
        predicted_probs = np.array(predicted_probs)

        # 计算微平均ROC曲线和AUC
        fpr, tpr, _ = roc_curve(true_labels_binary.ravel(), predicted_probs.ravel())
        roc_auc = auc(fpr, tpr)

        # 计算每个类别的ROC曲线和AUC
        class_auc_scores = []
        for i in range(num_classes):
            true_binary = (np.array(true_labels) == i).astype(int)
            pred_probs = predicted_probs[:, i]
            fpr_i, tpr_i, _ = roc_curve(true_binary, pred_probs)
            auc_score = auc(fpr_i, tpr_i)
            class_auc_scores.append(auc_score)
        roc_auc_one_vs_rest = np.mean(class_auc_scores)

        # 绘制并保存微平均ROC曲线
        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label='Micro-Average ROC curve (area = {0:0.4f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Micro-Average Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_curve_path = os.path.join(target_dir, f"{os.path.basename(dataset_path)}_roc_curve.png")
        plt.savefig(roc_curve_path)
        plt.close()

        # 绘制并保存单类别对其他类别平均的ROC曲线
        plt.figure()
        plt.plot(range(num_classes), class_auc_scores, color='blue', lw=2, label='One-vs-All ROC curve (area = {0:0.4f})'.format(roc_auc_one_vs_rest))
        plt.plot([0, num_classes], [0.5, 0.5], 'k--', lw=2)
        plt.xlim([0, num_classes])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Class')
        plt.ylabel('AUC Score')
        plt.title('One-vs-All Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        one_vs_all_roc_curve_path = os.path.join(target_dir, f"{os.path.basename(dataset_path)}_one_vs_all_roc_curve.png")
        plt.savefig(one_vs_all_roc_curve_path)
        plt.close()

        # 计算精确度、召回率和F1分数
        precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='micro')

        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, predicted_labels)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (fp + fn + tp)

        # 保存评估指标
        metrics_filename = f"{os.path.basename(dataset_path)}_metrics.txt"
        metrics_path = os.path.join(target_dir, metrics_filename)



        with open(metrics_path, "w") as f:
            f.write(f"Micro-Average AUC: {roc_auc:.5f}\n")
            f.write(f"One-vs-All Average AUC: {roc_auc_one_vs_rest:.5f}\n")
            f.write(f"Precision (micro-average): {precision:.5f}\n")
            f.write(f"Recall (micro-average): {recall:.5f}\n")
            f.write(f"F1 Score (micro-average): {f1_score:.5f}\n")
            f.write(f"True Positives (per class): {tp.tolist()}\n")
            f.write(f"False Positives (per class): {fp.tolist()}\n")
            f.write(f"False Negatives (per class): {fn.tolist()}\n")
            f.write(f"True Negatives (per class): {tn.tolist()}\n")

        print(f"Metrics saved to {metrics_path}")



# Usage example
model_config_resnet = {
    'name': 'ResNet50'
}
evaluator = ImageModelEvaluator(model_config_resnet)
dataset_paths = ["data/image/test_dataset", "data/image/test_dataset_perturbation_gaussian_noise_3"]
target_dir = "data/results"
evaluator.evaluate_and_save_metrics(dataset_paths, target_dir)
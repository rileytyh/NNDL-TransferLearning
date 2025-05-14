# NNDL-TransferLearning
**Project title:** *Detecting Novel Birds, Dogs, and Reptiles – Reliable OOD Detection via Energy & Softmax Thresholds*  
**COMS W4995 Neural Networks and Deep Learning**  
**Columbia University · Spring 2025**


## 📝 Overview  
We tackle hierarchical image classification in an **open-set** regime using the Bird–Dog–Reptile benchmark.  
A single frozen backbone (AlexNet, ResNet-V2, ConvNeXt-V2, or ViT-Huge/CLIP) feeds two linear heads (super- and sub-class).  
Novel examples are rejected with a simple confidence threshold (soft-max or energy).

## 📂 Files Submitted  
| Notebook | Description |
|----------|-------------|
| `W4995_TransferLearning_Project_AlexNet.ipynb` | Baseline AlexNet implementation and fine-tuning experiments |
| `W4995_TransferLearning_Project_ResNet.ipynb` | ResNet-V2 model and softmax-based thresholding |
| `W4995_TransferLearning_Project_ConvNeXt.ipynb` | ConvNeXt-V2 experiments for classification and OOD detection |
| `W4995_TransferLearning_Project_VIT.ipynb` | ViT-Huge CLIP backbone with hand-tuned thresholds for best performance |

## ▶️ How to Run  
1. **Environment**
   - Python ≥ 3.8
   - Required packages:
     ```bash
     pip install torch torchvision timm scikit-learn pandas matplotlib
     ```

2. **Data Setup**
   - Download and unzip the `Released_Data_NNDL_2025.zip` dataset into the working directory.
   - Folder structure:
     ```
      Released_Data_NNDL_2025/
      ├── train_data.csv
      ├── train_images/
      ├── test_images/
      ├── subclass_mapping.csv
      ├── superclass_mapping.csv
      ├── example_test_predictions.csv
      ├── simple_cnn_demo.ipynb
     ```

3. **Execution**
   - Open each notebook in order of interest (no strict dependency).
   - Each notebook trains and evaluates one model independently.
   - ViT notebook produces final leaderboard predictions.

## Final Results Summary (ViT)
- **Superclass Accuracy**: 97.33%
- **Subclass Accuracy**: 75.67%
- **AUROC**: 0.98 (superclass), 0.95 (subclass)

## Report
See our full write-up submitted as a separate PDF.

## Authors
- Tejas Chhotulal Badgujar
- Drew Levin
- Yuhao Tang
  
## Acknowledgments
Thanks to the COMS 4995 staff for providing the dataset, leaderboard, and support.

---


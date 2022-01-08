# VGG-Based auto encoder

---

This an implementation of a VGG-Based auto-encoder. The dataset used is bird species. The problem we aim to solve here is a classification problem. The issue we faced is that the number of training data is very limited. We tried to solve this problem by augmenting the data first, and then use a big autoencoder to capture the main characteristics of each class. Then use only the encoder part and fine-tune the encoder part only.

To augment the data, run the following command : 

```bash
python3 augment_v2.py --root-folder="birds_dataset/train_images"
```

To run the code : 

```bash
python3 main.py <data_directory>
```


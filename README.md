# 🥔 Potato Leaf Disease Classifier

A simple deep learning app built with PyTorch and Gradio to identify potato leaf diseases. Upload an image of a potato leaf, and the model will classify it as either:

* **Early Blight**
* **Late Blight**
* **Healthy**

Dataset: [Potato Leaf Disease Dataset](https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset)

## ⚠️ Notes

* The model file isn't included in this repository. You can train your own model or download a pre-trained one.

## 🧠 Model

The model is a custom Convolutional Neural Network (CNN) trained to classify potato leaf images into three categories:

* `Potato___Early_blight`
* `Potato___Late_blight`
* `Potato___Healthy`

Model input is resized to **224x224** and normalized with ImageNet statistics.

**Architecture Overview:**

* 3 Convolutional layers
* MaxPooling
* 2 Fully connected layers
* Log-Softmax output for 3-class prediction

> 🗂️ Model weights are expected to be in the same directory as `potato_model.pth`.

## 🧪 How to Use

1. Clone the repository.
2. Install dependencies:

```bash
pip install torch torchvision gradio pillow
```

3. Place the `potato_model.pth` file in the same directory.
4. Run the app:

```bash
python app.py
```

5. Upload a potato leaf image and get predictions with confidence scores.

## 📁 File Structure

```
├── app.py              # Main application script
├── potato_model.pth    # Trained model weights (not included)
└── README.md           # This file
```
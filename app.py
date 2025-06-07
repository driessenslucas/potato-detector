import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___Healthy']

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406],  # Normalize with ImageNet means and stds
                         [0.229, 0.224, 0.225])
])


class PotoatoModel(nn.Module):
    def __init__(self):
        super(PotoatoModel, self).__init__()
        # Define the CNN architecture that processes the 3 classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 3)
        
    def forward(self, x):
        # Forward pass through the CNN
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Output layer for 3 classes
        x = F.log_softmax(x, dim=1)
        
        return x

try:
    model = PotoatoModel()
    model.load_state_dict(torch.load('potato_model.pth', map_location=device))
    
    model.to(device)
    print(f"Model loaded successfully on {device}")
except FileNotFoundError:
    print("Error: potato_model.pth not found. Make sure the model file is in the correct directory.")
    exit(1)

def predict_image(img):
    """
    Predict potato leaf disease from an uploaded image
    
    Args:
        img: PIL Image object
    
    Returns:
        tuple: (prediction_label, confidence_percentage)
    """
    if img is None:
        return "No image provided", "0.00%"
    
    try:
        model.eval()
        
        # Ensure image is RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Transform and add batch dimension
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        
        # Format class name for better display
        
        # print % of confidence for each class
        print(f"Confidence: {conf.item()*100:.2f}% for class index {pred_idx.item()}")
        class_name = class_names[pred_idx.item()]
        
        formatted_name = class_name.replace('Potato___', '').replace('_', ' ').title()
        
        return formatted_name, f"{conf.item()*100:.2f}%"
    
    except Exception as e:
        return f"Error: {str(e)}", "0.00%"

# Create Gradio Interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Potato Leaf Image"),
    outputs=[
        gr.Textbox(label="Prediction", show_label=True),
        gr.Textbox(label="Confidence", show_label=True)
    ],
    title="🥔 Potato Leaf Disease Classifier",
    description="Upload an image of a potato leaf to identify potential diseases. The model can detect Early Blight, Late Blight, or classify leaves as Healthy.",
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False
    )
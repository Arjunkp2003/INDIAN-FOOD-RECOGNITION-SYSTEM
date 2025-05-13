# INDIAN-FOOD-RECOGNITION-SYSTEM

I’m excited to share my Indian Food Recognition System, a web app that celebrates India’s culinary diversity through AI. This system classifies 80 Indian dishes—from Biryani to Adhirasam—using a fine-tuned MobileNetV2 model for feature extraction and a Random Forest classifier for accurate predictions. Built with Python, PyTorch, Gradio, and SQLite, it offers secure user authentication, real-time predictions with top-5 confidence scores, and a responsive interface styled with modern CSS. Images are preprocessed with torchvision transforms and displayed with predictions overlaid, creating an engaging experience. Designed for local deployment, the app ensures easy access for users to explore Indian cuisine. This project blends my passion for coding and India’s vibrant food culture, educating users about its diverse flavors. Future enhancements include recipe suggestions, an expanded dataset, and cloud deployment for broader reach. Check out the code [GitHub link, if applicable] and connect with me to discuss AI, food, or potential collaborations! I’d love to hear your feedback on this fusion of technology and culinary arts. #MachineLearning #IndianCuisine #AI #Python #Gradio





#Indian Food Recognition System
#1. Key Features & Technologies Used
#Key Features:

Multi-Class Classification: Identifies 80 Indian dishes, from Biryani to Adhirasam, with top-5 predictions and confidence scores.
User Authentication: Secure registration and login system using SQLite with SHA-256 password hashing.
Real-Time Predictions: Processes uploaded images instantly, displaying results with a visually appealing overlay.
Interactive UI: Responsive and intuitive interface with smooth navigation between home, login, registration, and prediction pages.
Responsive Design: Styled with modern CSS, featuring background images and a polished, user-friendly layout.

#Technologies Used:

Model Architecture: Fine-tuned MobileNetV2 for feature extraction, paired with a Random Forest classifier.
Tech Stack: Python, PyTorch, Gradio, SQLite, Scikit-learn, Matplotlib, PIL, torchvision.
Data Preprocessing: Image resizing, normalization, and transformation using torchvision.
Deployment: Local deployment via Gradio for seamless user access.
Environment: Supports CPU/GPU inference, optimized for performance.

#2. Workflow (Step-by-Step)
1. Dataset Preparation:

Collect and preprocess images of 80 Indian dishes.
Apply torchvision transforms (resize to 224x224, normalize) for model compatibility.

2. Model Setup:

Load pretrained MobileNetV2 and modify classifier for 80 classes.
Extract features using MobileNetV2, then train Random Forest on features.

3. Environment Setup:

Install dependencies (PyTorch, Gradio, Scikit-learn) in a Python environment.
Initialize SQLite database for user authentication.

4. Training & Integration:

Fine-tune MobileNetV2 on custom dataset (weights saved as mobilenet_indian_food.pth).
Train Random Forest classifier (saved as random_forest_classifier.pkl).
Integrate models into Gradio for real-time predictions.

5. User Interface Development:

Design Gradio interface with CSS for home, login, register, and predict pages.
Implement navigation logic and user session management.

6. Prediction Pipeline:

User uploads image via Gradio.
Image is preprocessed, features extracted via MobileNetV2, and classified by Random Forest.
Display predicted class, confidence, top-5 predictions, and annotated image.

7. Deployment:

Launch app locally using demo.launch() in Gradio.
Test authentication, navigation, and prediction functionality.

This project blends AI with India’s culinary diversity, offering a robust and engaging user experience. Connect to discuss or explore the code [GitHub link, if applicable]! #MachineLearning #IndianCuisine #AI #Python #Gradio

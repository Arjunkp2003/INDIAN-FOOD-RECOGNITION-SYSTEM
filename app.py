import gradio as gr
import sqlite3
import hashlib
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import joblib
import matplotlib.pyplot as plt
import io
import base64

# ---------- Setup SQLite ----------
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
""")
conn.commit()

# ---------- Helper Functions for Authentication ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, confirm_password):
    if not username or not password or not confirm_password:
        return "All fields are required."
    if password != confirm_password:
        return "Passwords do not match."
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                       (username, hash_password(password)))
        conn.commit()
        return "Registration successful. You can now login."
    except sqlite3.IntegrityError:
        return "Username already exists."

def login_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", 
                   (username, hash_password(password)))
    return cursor.fetchone() is not None

# ---------- Setup Prediction Model ----------
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MobileNetV2 model
num_classes = 80
mobilenet = models.mobilenet_v2(pretrained=False)
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, num_classes)
mobilenet.load_state_dict(torch.load("mobilenet_indian_food.pth", map_location=device))
mobilenet.eval()
mobilenet.to(device)

# Load Random Forest model
rf = joblib.load("random_forest_classifier.pkl")

# Class mapping
class_to_idx = {
    "adhirasam": 0, "aloo_gobi": 1, "aloo_matar": 2, "aloo_methi": 3, "aloo_shimla_mirch": 4,
    "aloo_tikki": 5, "anarsa": 6, "ariselu": 7, "bandar_laddu": 8, "basundi": 9,
    "bhatura": 10, "bhindi_masala": 11, "biryani": 12, "boondi": 13, "butter_chicken": 14,
    "chak_hao_kheer": 15, "cham_cham": 16, "chana_masala": 17, "chapati": 18, "chhena_kheeri": 19,
    "chicken_razala": 20, "chicken_tikka": 21, "chicken_tikka_masala": 22, "chikki": 23,
    "daal_baati_churma": 24, "daal_puri": 25, "dal_makhani": 26, "dal_tadka": 27,
    "dharwad_pedha": 28, "doodhpak": 29, "double_ka_meetha": 30, "dum_aloo": 31,
    "gajar_ka_halwa": 32, "gavvalu": 33, "ghevar": 34, "gulab_jamun": 35, "imarti": 36,
    "jalebi": 37, "kachori": 38, "kadai_paneer": 39, "kadhi_pakoda": 40, "kajjikaya": 41,
    "kakinada_khaja": 42, "kalakand": 43, "karela_bharta": 44, "kofta": 45, "kuzhi_paniyaram": 46,
    "lassi": 47, "ledikeni": 48, "litti_chokha": 49, "lyangcha": 50, "maach_jhol": 51,
    "makki_di_roti_sarson_da_saag": 52, "malapua": 53, "misi_roti": 54, "misti_doi": 55,
    "modak": 56, "mysore_pak": 57, "naan": 58, "navrattan_korma": 59, "palak_paneer": 60,
    "paneer_butter_masala": 61, "phirni": 62, "pithe": 63, "poha": 64, "poornalu": 65,
    "pootharekulu": 66, "qubani_ka_meetha": 67, "rabri": 68, "rasgulla": 69, "ras_malai": 70,
    "sandesh": 71, "shankarpali": 72, "sheera": 73, "sheer_korma": 74, "shrikhand": 75,
    "sohan_halwa": 76, "sohan_papdi": 77, "sutar_feni": 78, "unni_appam": 79
}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature extraction for a single image
def extract_features_single_image(image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        features = mobilenet.features(image_tensor)
        features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        features = features.view(features.size(0), -1)
    return features.cpu().numpy()

# Predict single image
def predict_image(image: Image.Image):
    try:
        # Preprocess image
        image = image.convert('RGB')
        image_tensor = transform(image)

        # Extract features
        features = extract_features_single_image(image_tensor)

        # Predict with Random Forest
        pred_label = rf.predict(features)[0]
        pred_prob = rf.predict_proba(features)[0]

        # Get results
        predicted_class = idx_to_class[pred_label]
        confidence = np.max(pred_prob)

        # Top-5 predictions
        top5_indices = np.argsort(pred_prob)[::-1][:5]
        top5_text = "\nTop 5 Predictions:\n" + "\n".join(
            [f"{idx_to_class[i]}: {pred_prob[i]:.4f}" for i in top5_indices]
        )

        # Create image with prediction
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"Predicted: {predicted_class}")
        plt.axis('off')
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        img_html = f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.2);"/>'

        # Return formatted prediction
        prediction_text = (
            f"**Predicted Class**: {predicted_class}\n"
            f"**Confidence**: {confidence:.4f}\n"
            f"{top5_text}"
        )
        return prediction_text, img_html

    except Exception as e:
        return f"Error processing image: {str(e)}", None

# ---------- Page Navigation States ----------
current_user = {"username": None}

# ---------- Gradio Interface ----------
with gr.Blocks(css="""
.gradio-container {
    background-image: url('home.jpg');
    background-size: cover;
    background-position: center;
    height: 100vh;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.home-page, .login-page, .register-page, .predict-page {
    background: rgba(255, 255, 255, 0.95);
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    max-width: 700px;
    margin: 50px auto;
    text-align: center;
    backdrop-filter: blur(10px);
}

h2, h3 {
    font-family: 'Segoe UI Semibold', sans-serif;
    font-size: 2em;
    color: #2c3e50;
}

p, label {
    font-size: 1.1em;
    color: #34495e;
    line-height: 1.6em;
}

.markdown {
    font-size: 1.05em;
    color: #2c3e50;
}

button {
    background: linear-gradient(to right, #ff512f, #dd2476);
    color: white;
    border: none;
    padding: 14px 28px;
    font-size: 1em;
    font-weight: bold;
    cursor: pointer;
    border-radius: 10px;
    transition: all 0.3s ease-in-out;
    margin: 10px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
}

button:hover {
    background: linear-gradient(to right, #dd2476, #ff512f);
    transform: scale(1.03);
}

input[type="text"], input[type="password"], textarea {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 12px;
    font-size: 1em;
    margin-top: 5px;
    width: 100%;
    box-sizing: border-box;
    transition: border 0.3s ease;
}

input[type="text"]:focus, input[type="password"]:focus {
    border: 1px solid #ff512f;
    outline: none;
}

.gr-image {
    border-radius: 12px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}
""") as demo:

    # Home Page
    with gr.Column(visible=True, elem_id="home-page", elem_classes=["home-page"]) as home_page:
        gr.Image(value="home.jpg", show_label=False, show_download_button=False)
        with gr.Group():
            gr.Markdown("# 🍽️ Welcome to Indian Foods Recognition System !", elem_classes=["markdown"])
            gr.Markdown("""
            ## About Indian Foods
            Indian cuisine is known for its rich flavors, diverse ingredients, and the use of a variety of spices.  
            From the aromatic biryanis of the north to the spicy curries of the south,  
            Indian food offers an amazing variety of tastes and textures.  
            Whether it's vegetarian or non-vegetarian, Indian food caters to all tastes and preferences.  
            It is a celebration of culture, history, and the unique regional influences that make every dish special.
            """, elem_classes=["markdown"])
            with gr.Row():
                go_login_btn = gr.Button("Go to Login")
                go_register_btn = gr.Button("Go to Register")

    # Login Page
    with gr.Column(visible=False, elem_id="login-page") as login_page:
        gr.Image(value="register.jpg", show_label=False, show_download_button=False)
        gr.Markdown("## 🔐 Login")
        login_username = gr.Textbox(label="Username")
        login_password = gr.Textbox(label="Password", type="password")
        login_status = gr.Textbox(label="Status", interactive=False)
        login_btn = gr.Button("Login")
        go_register_btn2 = gr.Button("Don't have an account? Register here")
        back_home_btn_login = gr.Button("Back to Home")

    # Register Page
    with gr.Column(visible=False, elem_id="register-page") as register_page:
        gr.Image(value="register.jpg", show_label=False, show_download_button=False)
        gr.Markdown("## 📝 Register")
        reg_username = gr.Textbox(label="Username")
        reg_password = gr.Textbox(label="Password", type="password")
        reg_confirm = gr.Textbox(label="Confirm Password", type="password")
        reg_status = gr.Textbox(label="Status", interactive=False)
        reg_btn = gr.Button("Register")
        back_to_login_btn = gr.Button("Back to Login")
        back_home_btn_register = gr.Button("Back to Home")

    # Predict Page
    with gr.Column(visible=False, elem_id="predict-page") as predict_page:
        welcome_text = gr.Markdown("## 🍛 Hello, User")
        gr.Markdown("Upload an Indian food image to classify it!")
        image_input = gr.Image(type="pil", show_label=False, interactive=True)
        prediction_output = gr.Markdown(label="Prediction")
        predicted_image = gr.HTML(label="Predicted Image")
        logout_btn = gr.Button("Logout")

    # ---------- Page Navigation Logic ----------
    def show_login():
        return (gr.update(visible=False), gr.update(visible=True),
                gr.update(visible=False), gr.update(visible=False))

    def show_register():
        return (gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=False))

    def show_predict():
        return (gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=True))

    def show_home():
        return (gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False))

    def handle_login(u, p):
        if login_user(u, p):
            current_user["username"] = u
            return gr.update(value=f"## 🍛 Hello, {u}"), "", *show_predict()
        else:
            return gr.update(value="## 🔐 Login"), "Invalid credentials", *show_login()

    def handle_register(u, p1, p2):
        return register_user(u, p1, p2)

    def logout():
        current_user["username"] = None
        return gr.update(value="## 🔐 Login"), "", None, *show_login()

    # ---------- Button Click Bindings ----------
    go_login_btn.click(fn=show_login, outputs=[home_page, login_page, register_page, predict_page])
    go_register_btn.click(fn=show_register, outputs=[home_page, login_page, register_page, predict_page])
    back_to_login_btn.click(fn=show_login, outputs=[home_page, login_page, register_page, predict_page])
    back_home_btn_login.click(fn=show_home, outputs=[home_page, login_page, register_page, predict_page])
    back_home_btn_register.click(fn=show_home, outputs=[home_page, login_page, register_page, predict_page])
    go_register_btn2.click(fn=show_register, outputs=[home_page, login_page, register_page, predict_page])
    login_btn.click(fn=handle_login, inputs=[login_username, login_password],
                    outputs=[welcome_text, login_status, home_page, login_page, register_page, predict_page])
    reg_btn.click(fn=handle_register, inputs=[reg_username, reg_password, reg_confirm],
                  outputs=[reg_status])
    logout_btn.click(fn=logout, outputs=[welcome_text, login_status, predicted_image, home_page, login_page, register_page, predict_page])
    image_input.change(fn=predict_image, inputs=image_input, outputs=[prediction_output, predicted_image])

# ---------- Launch App ----------
if __name__ == "__main__":
    demo.launch()
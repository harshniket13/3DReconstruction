from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from torchvision import transforms
import open3d as o3d
# from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2Tokenizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load MiDaS model for depth estimation
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to('cpu').eval()

# Load BLIP processor and model for image captioning
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2Tokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def process_image(file_path):
    # Load and preprocess the image
    image = Image.open(file_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Resize to fixed dimensions
        transforms.ToTensor()
    ])
    input_image = transform(image).unsqueeze(0)

    # Predict depth
    with torch.no_grad():
        prediction = midas(input_image)
    depth = prediction.squeeze().numpy()

    # Resize image to match depth map dimensions
    image = image.resize((depth.shape[1], depth.shape[0]))
    
    # Save depth image
    depth_image_path = os.path.join('static/results', 'depth_image.png')
    plt.imsave(depth_image_path, depth, cmap='plasma')

    # Generate 3D point cloud
    h, w = depth.shape
    fx, fy = w / 2.0, h / 2.0
    cx, cy = w / 2.0, h / 2.0
    points = []
    colors = []
    for v in range(h):
        for u in range(w):
            z = depth[v, u]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])
            colors.append(image.getpixel((u, v)))

    points = np.array(points)
    colors = np.array(colors) / 255.0  # Normalize colors to [0, 1] range for Open3D

    # Create Open3D point cloud with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate normals for the point cloud
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Convert point cloud to mesh
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    # Assign original colors to the mesh vertices
    mesh.vertex_colors = pcd.colors

    # Export the mesh
    mesh_path = os.path.join('static/results/', 'mesh.ply')
    o3d.io.write_triangle_mesh(mesh_path, mesh)

    # Create point cloud plot
    point_cloud_path = os.path.join('static/results', 'point_cloud.html')
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers', marker=dict(size=2, color=colors)
    )])
    fig.write_html(point_cloud_path)
    
    # Generate brief text description using BLIP model
    # inputs = blip_processor(images=image, return_tensors="pt")
    # with torch.no_grad():
    #     out = blip_model.generate(**inputs)
    # description = blip_processor.decode(out[0], skip_special_tokens=True)
    
    # Preprocess the image
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    # Generate description
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        description = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
    print("\nGenerated Description:", description)

    return depth_image_path, point_cloud_path, mesh_path, description

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            depth_image_path, point_cloud_path, mesh_path, description = process_image(file_path)
            return render_template('index.html', image_path=file_path, depth_image_path=depth_image_path, point_cloud_path=point_cloud_path, mesh_path=mesh_path, description=description)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)

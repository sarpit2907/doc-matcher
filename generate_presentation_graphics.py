import sys
import os
from pathlib import Path
import json
import torch
import cv2
import numpy as np

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from line_matching.line_lightglue.line_lightglue_model import LineLightGlue
from line_matching.line_lightglue.graph_transformer import GraphVisualization

def main():
    print("Initializing GNN Architecture internally...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    gnn_conf = {
        "use_graph_transformer": True,
        "graph_k_neighbors": 5,
        "graph_edge_dim": 4,
        "graph_sparse_attention": False,
    }
    
    model = LineLightGlue(gnn_conf).to(device)
    
    print("Synthesizing controlled Document Layout...")
    # Synthesize a clean 3x3 table with a header
    lines_data = []

    # Document Header
    lines_data.append({"polygon": [[200, 50], [300, 50], [300, 80], [200, 80]]})

    # 3x3 Table layout
    for row in range(3):
        for col in range(3):
            x = 50 + col * 150
            y = 150 + row * 80
            lines_data.append({"polygon": [[x, y], [x+100, y], [x+100, y+20], [x, y+20]]})

    N = len(lines_data)
    descriptors = torch.randn(1, N, 256)
    
    # Inject spatial coordinates so the K-NN graph perfectly reflects geography
    for i, line in enumerate(lines_data):
        cx = np.mean([p[0] for p in line["polygon"]])
        cy = np.mean([p[1] for p in line["polygon"]])
        descriptors[0, i, 0] = cx * 10.0
        descriptors[0, i, 1] = cy * 10.0

    descriptors = descriptors.to(device)

    print("Running GNN Forward Pass...")
    with torch.no_grad():
        _ = model.transformers[0].self_attn(descriptors)  
        graph_info = model.transformers[0].self_attn.get_graph_info()
        adj_matrix = graph_info['adj_matrix'][0]

    # Draw blank canvas
    blank_doc = np.ones((500, 500, 3), dtype=np.uint8) * 255
    cv2.imwrite("blank_doc.jpg", blank_doc)

    out_path = "Presentation_Graphic.jpg"
    print(f"Plotting Graph onto {out_path}...")
    
    GraphVisualization.plot_graph_on_image(
        image_path="blank_doc.jpg",
        lines_data=lines_data,
        adj_matrix=adj_matrix,
        output_path=out_path,
        title="DocMatcher GNN K-Nearest Neighbor Edge Connectivity",
        figsize=(8, 8)
    )
    
    # Cleanup empty canvas
    if os.path.exists("blank_doc.jpg"):
        os.remove("blank_doc.jpg")
        
    print(f"DONE! Please find {out_path} in your file manager.")

if __name__ == "__main__":
    main()

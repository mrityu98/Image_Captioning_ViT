"""
Image Captioning Web App
Swin Transformer + GPT-2 Model
"""

import os
os.environ['STREAMLIT_WATCHER_TYPE'] = 'none'

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import timm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchvision import transforms

st.set_page_config(
    page_title="Image Captioning",
    page_icon="🖼️",
    layout="centered"
)

class ImageCaptioningModel(nn.Module):
    def __init__(self, swin_model_name, gpt2_model_name, tokenizer):
        super(ImageCaptioningModel, self).__init__()
        
        self.tokenizer = tokenizer
        self.swin_embed_dim = 768
        self.gpt2_embed_dim = 768
        
        # Swin Transformer
        self.swin = timm.create_model(
            swin_model_name,
            pretrained=False,
            num_classes=0
        )
        
        # Freeze Swin
        for param in self.swin.parameters():
            param.requires_grad = False
        
        # Projection layer
        self.projection = nn.Linear(self.swin_embed_dim, self.gpt2_embed_dim)
        
        # GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.gpt2.resize_token_embeddings(len(tokenizer))
        
        self.image_prefix_length = 1
    
    def generate_caption(self, image, max_length=30, device='cpu'):
        """Generate caption for a single image"""
        self.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(device)
            
            # Extract features
            image_features = self.swin(image)
            image_embeds = self.projection(image_features).unsqueeze(1)
            
            # Start with BOS token
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=device)
            
            # Generate tokens
            for _ in range(max_length):
                caption_embeds = self.gpt2.transformer.wte(input_ids)
                input_embeds = torch.cat([image_embeds, caption_embeds], dim=1)
                outputs = self.gpt2(inputs_embeds=input_embeds, return_dict=True)
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token_id], dim=1)
                
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break
            
            caption = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        return caption


@st.cache_resource
def load_model():
    # Download model from Hugging Face
    token = st.secrets.get("HF_TOKEN", None)
    
    model_path = hf_hub_download(
        repo_id="digital-base/SWIN-GPT-Image_Caption",
        filename="best_model.pt",
        token=token,
        cache_dir="./model_cache"
    )
    
    # Load the model
    model = torch.load(model_path, map_location='cpu')
    return model

def preprocess_image(image):
    """Preprocess image for model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image)


def main():
    # Title
    st.title("🖼️ Image Captioning")
    st.markdown("Generate captions for your images using **Swin Transformer + GPT-2**")
    
    # Load model
    model, tokenizer, device = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    st.markdown("### Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a JPG, JPEG, or PNG image"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.markdown("### Generated Caption")
            
            # Generate caption button
            if st.button("🎯 Generate Caption", type="primary"):
                with st.spinner("Generating caption..."):
                    # Preprocess
                    img_tensor = preprocess_image(image)
                    
                    # Generate
                    caption = model.generate_caption(img_tensor, max_length=30, device=device)
                    
                    # Display
                    st.markdown("**Caption:**")
                    st.success(caption)
                    
                    st.balloons()
    
    else:
        # Show example
        st.info("👆 Upload an image to get started!")
        
        st.markdown("### Example")
        st.markdown("Upload an image and click 'Generate Caption' to see the magic! ✨")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from torchvision import models, transforms
import streamlit as st
import torch
import pickle
from PIL import Image
#from io import BytesIO
#import ssl

#ssl._create_default_https_context = ssl._create_unverified_context

def predict(image_path):
    resnet = models.resnet101(pretrained=True)
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    resnet.eval()
    out = resnet(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title('Image Detection Algorithm')
    st.subheader("Let us tell you what that image is")
    file_up = st.file_uploader("Upload a single image, only JPEG", type="jpeg")
    if file_up:
        image = Image.open(file_up)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        st.write("Just a second...")
        labels = predict(file_up)
        #st.write(labels)
        scores = []
        item = []


        for i in labels:
            st.write("It can be a ", i[0][4:], " with a certainty of  ",np.round( i[1],decimals=3), '%')
            item.append(i[0][4:])
            scores.append(i[1])
        max_score = max(scores)
        max_item = item[scores.index(max_score)]

        #st.write(max_item)
        #st.write(str(scores))
        st.subheader('Most Likely, the image contain a  ' +str(max_item) + '.')

if __name__ == '__main__':
     main()

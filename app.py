import streamlit as st
from fastai.vision.all import *
import pathlib
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

#title
st.title("Transportni aniqlovchi model")

#file joylash
file=st.file_uploader("Rasm yuklash", type=['png', 'jpg'])
if file:
    st.image(file)

    #pil convert
    img=PILImage.create(file)

    #modelni chaqirish
    model=load_learner('transport.pkl')

    #predict
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

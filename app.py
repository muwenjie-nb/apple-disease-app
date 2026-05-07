import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="苹果叶病识别系统", layout="centered")
st.title("🍎 苹果叶病识别系统（演示版）")
st.markdown("> ⚠️ 科研演示版本：识别结果仅供参考，置信度通常在30-60%之间")

name_mapping = {
    "cedar rust": "苹果雪松锈病",
    "apple scab": "苹果黑星病",
    "apple black rot": "苹果黑腐病",
}

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.markdown("**📸 操作说明：点击「Browse files」，选择苹果叶片照片上传**")

uploaded_file = st.file_uploader("上传苹果叶片照片", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", width=300)
    
    with st.spinner("识别中..."):
        results = model(image, conf=0.25)
    
    result = results[0]
    boxes = result.boxes
    
    if boxes is not None and len(boxes) > 0:
        best_conf = 0
        best_cls_id = None
        for box in boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_cls_id = int(box.cls[0])
        
        class_name_en = model.names[best_cls_id]
        class_name_cn = name_mapping.get(class_name_en, f"未知病害（英文：{class_name_en}）")
        
        st.success(f"**预测结果**：{class_name_cn}")
        st.write(f"**置信度**：{best_conf:.1%}")
        
        if best_conf < 0.4:
            st.info("💡 置信度偏低，建议拍摄更清晰、病斑更明显的叶片特写重新识别")
    else:
        st.info("🍃 未检测到明显病斑，叶片可能健康")

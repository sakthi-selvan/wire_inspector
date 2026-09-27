import hashlib
import io
import json
import time
import zipfile
import streamlit as st
from engine import Inspector, MODEL_ID, LABELS, read_image, summarize, png_bytes

st.set_page_config(page_title='Wire Inspector', page_icon='🔎', layout='wide')
st.title('Wire Inspector')
st.caption('Upload a wire image • Find possible defects • Inspect segmentation masks')
st.info('Experimental pretrained AI. Masks are approximate; burnmark, cut, crack and scratch predictions require visual review.')

@st.cache_resource(show_spinner=False)
def load_model():
    return Inspector()

with st.sidebar:
    st.header('Inspection settings')
    threshold = st.slider('Defect mask threshold', 0.1, 0.9, 0.5, 0.05)
    wire_threshold = st.slider('Wire region threshold', 0.1, 0.9, 0.35, 0.05)
    min_pixels = st.number_input('Minimum region size (pixels)', 1, 10000, 30)
    st.caption('Lower thresholds highlight more areas and may create false alarms. Activations are not accuracy or confidence percentages.')
    st.markdown('**Model:** CLIPSeg RD64 refined')
    st.caption('Runs locally. First use needs internet to download model weights (several hundred MB). CPU supported; GPU optional.')

upload = st.file_uploader('Upload a close-up wire photo', type=['jpg', 'jpeg', 'png', 'webp'])
if upload is None:
    st.write('Use a sharp photo with the wire filling most of the image and a simple background.')
    st.stop()
try:
    raw = upload.getvalue()
    image = read_image(raw)
except Exception as exc:
    st.error(f'Cannot read this image: {exc}')
    st.stop()
key = hashlib.sha256(raw).hexdigest()
if st.session_state.get('image_key') != key:
    st.session_state.pop('prediction', None)
    st.session_state['image_key'] = key
left, right = st.columns(2)
left.image(image, caption='Uploaded image', use_container_width=True)
if st.button('Analyze wire', type='primary'):
    try:
        with st.spinner('Loading model and inspecting image… first run takes longer.'):
            model = load_model()
            start = time.perf_counter()
            st.session_state['prediction'] = model.predict(image)
            st.session_state['elapsed'] = time.perf_counter() - start
            st.session_state['device'] = model.device
    except Exception as exc:
        st.error('Analysis could not complete. Check internet access for the initial model download and available memory.')
        st.code(str(exc))

if 'prediction' in st.session_state:
    status, rows, overlay, mask = summarize(image, st.session_state['prediction'],
                                          threshold, wire_threshold, int(min_pixels))
    right.image(overlay, caption='Candidate defect segmentation', use_container_width=True)
    st.subheader(status)
    st.caption(f"Inference: {st.session_state['elapsed']:.1f}s • {st.session_state['device'].upper()} • Processed size: {image.width} × {image.height}")
    st.markdown('🟥 Burnmark · 🟨 Cut · 🟪 Crack · 🟦 Scratch')
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.write('No highlighted defect regions. This does not establish that the wire is defect-free.')
    with st.expander('View raw activation maps'):
        columns = st.columns(4)
        for i, label in enumerate(LABELS):
            columns[i].image(st.session_state['prediction'][i], clamp=True, caption=label, use_container_width=True)
    report = {'model': MODEL_ID, 'status': status, 'results': rows,
              'processed_size': [image.width, image.height],
              'settings': {'defect_threshold': threshold, 'wire_threshold': wire_threshold,
                           'minimum_region_pixels': int(min_pixels)},
              'mask_values': {'0': 'background/unassigned', '1': 'Burnmark', '2': 'Cut', '3': 'Crack', '4': 'Scratch'},
              'note': 'Unvalidated zero-shot prototype. Activations are not calibrated probabilities. Mask coordinates refer to processed image.'}
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('overlay.png', png_bytes(overlay))
        archive.writestr('mask.png', png_bytes(mask))
        archive.writestr('report.json', json.dumps(report, indent=2))
    st.download_button('Download results', data.getvalue(), 'wire_inspection_results.zip', 'application/zip')

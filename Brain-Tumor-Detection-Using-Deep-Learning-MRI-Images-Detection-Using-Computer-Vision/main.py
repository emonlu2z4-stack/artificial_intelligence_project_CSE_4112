import os
import time
import uuid
from datetime import datetime
import numpy as np
from flask import Flask, request, send_from_directory, jsonify, url_for, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import jinja2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    static_folder=BASE_DIR,
    template_folder=os.path.join(BASE_DIR, 'templates')
)
app.jinja_loader = jinja2.ChoiceLoader([
    jinja2.FileSystemLoader(os.path.join(BASE_DIR, 'templates')),
    jinja2.FileSystemLoader(BASE_DIR)
])
app.secret_key = 'neuroscan_ai_secret_key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

CORS(app, resources={r"/*": {"origins": "*"}})

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp'}

# নোটবুকের ফোল্ডার স্ট্রাকচার অনুযায়ী ক্লাস লেবেল
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

CLASS_METADATA = {
    'glioma': {
        'display_name': 'Glioma',
        'is_tumor': True,
        'severity': 'High Clinical Attention',
        'badge_class': 'danger',
        'color': '#ef4444',
        'description': 'Gliomas originate from glial cells and represent the most prevalent group of primary central nervous system neoplasms.',
        'recommendation': 'Immediate neuro-oncological evaluation and contrast-enhanced MRI (T1-Gd, T2/FLAIR) are recommended.'
    },
    'meningioma': {
        'display_name': 'Meningioma',
        'is_tumor': True,
        'severity': 'Moderate to High Attention',
        'badge_class': 'warning',
        'color': '#f59e0b',
        'description': 'Meningiomas arise from arachnoid cap cells within the meninges and are predominantly extra-axial and slow-growing.',
        'recommendation': 'Neurosurgical consultation recommended to evaluate mass effect and surgical resection options.'
    },
    'pituitary': {
        'display_name': 'Pituitary Tumor',
        'is_tumor': True,
        'severity': 'Specialist Evaluation Required',
        'badge_class': 'primary',
        'color': '#6366f1',
        'description': 'Pituitary tumors develop within the sella turcica and can cause endocrine imbalances or optic chiasm compression.',
        'recommendation': 'Comprehensive neuroendocrine hormonal blood panel and visual field perimetry testing.'
    },
    'notumor': {
        'display_name': 'No Tumor Detected',
        'is_tumor': False,
        'severity': 'Normal / Non-Neoplastic',
        'badge_class': 'success',
        'color': '#10b981',
        'description': 'No evident intracranial mass, pathological enhancement, or mass effect detected in this scan.',
        'recommendation': 'Negative for focal neoplasms. Correlate with clinical findings and radiologist evaluation.'
    }
}

MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
model = None
model_path_used = None

if os.path.exists(MODEL_PATH):
    try:
        print(f"Loading neural network model from: {MODEL_PATH}")
        model = load_model(MODEL_PATH, compile=False)
        model_path_used = MODEL_PATH
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"WARNING: '{MODEL_PATH}' file not found!")


def get_expected_image_size():
    """মডেলের ইনপুট শেপ চেক করে, না পেলে 128x128 নির্ধারণ করে"""
    if model is not None and hasattr(model, 'input_shape') and model.input_shape:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if len(shape) == 4 and shape[1] is not None and shape[2] is not None:
            return int(shape[1]), int(shape[2])
    return 128, 128


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_mri_image(image_path):
    """নোটবুকের ট্রেনিং অনুযায়ী ঠিক 128x128 সাইজে 0-1 স্কেলিং করে"""
    target_w, target_h = get_expected_image_size()

    with Image.open(image_path) as pil_img:
        pil_img = ImageOps.exif_transpose(pil_img)

        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        resample_filter = getattr(Image, 'Resampling', Image).BILINEAR
        pil_img = pil_img.resize((target_w, target_h), resample_filter)
        img_array = np.asarray(pil_img, dtype=np.float32) / 255.0
        return img_array


def predict_tumor(image_path):
    if model is None:
        raise RuntimeError("model.h5 is not loaded.")

    img_array = prepare_mri_image(image_path)
    input_tensor = np.expand_dims(img_array, axis=0)

    raw_preds = model.predict(input_tensor, verbose=0)[0]

    # সফটম্যাক্স ভ্যালু নিশ্চিত করা
    if np.any(raw_preds < 0) or not np.isclose(np.sum(raw_preds), 1.0, atol=1e-2):
        exp_preds = np.exp(raw_preds - np.max(raw_preds))
        raw_preds = exp_preds / np.sum(exp_preds)

    pred_idx = int(np.argmax(raw_preds))
    pred_label = CLASS_LABELS[pred_idx]
    confidence_val = float(raw_preds[pred_idx])

    meta = CLASS_METADATA.get(pred_label, CLASS_METADATA['notumor'])

    probabilities = []
    for idx, label in enumerate(CLASS_LABELS):
        m = CLASS_METADATA.get(label, {})
        prob_pct = float(raw_preds[idx]) * 100.0
        probabilities.append({
            'label': label,
            'display_name': m.get('display_name', label.capitalize()),
            'probability': round(prob_pct, 2),
            'probability_formatted': f"{prob_pct:.2f}%",
            'is_predicted': (idx == pred_idx),
            'badge_class': m.get('badge_class', 'secondary'),
            'color': m.get('color', '#0284c7')
        })

    probabilities_sorted = sorted(probabilities, key=lambda x: x['probability'], reverse=True)

    result_title = meta.get('display_name')
    if meta.get('is_tumor', False):
        result_title = f"Tumor Detected: {meta.get('display_name')}"

    return {
        'predicted_label': pred_label,
        'display_name': meta.get('display_name'),
        'is_tumor': meta.get('is_tumor', False),
        'result_title': result_title,
        'confidence': f"{confidence_val * 100.0:.2f}%",
        'confidence_raw': round(confidence_val * 100.0, 2),
        'severity': meta.get('severity', 'Diagnostic Assessment'),
        'badge_class': meta.get('badge_class', 'info'),
        'description': meta.get('description', ''),
        'recommendation': meta.get('recommendation', ''),
        'probabilities': probabilities_sorted,
        'scan_id': f"MRI-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        'analyzed_at': datetime.now().strftime('%b %d, %Y • %H:%M:%S')
    }


@app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except Exception:
        root_index = os.path.join(BASE_DIR, 'index.html')
        if os.path.exists(root_index):
            with open(root_index, 'r', encoding='utf-8') as f:
                return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
        raise


@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'model_path': model_path_used or 'None'
    })


@app.route('/api/sample/<filename>', methods=['GET'])
def api_sample(filename):
    safe_name = secure_filename(filename)
    path = os.path.join(BASE_DIR, safe_name)
    if not os.path.exists(path):
        return jsonify({'error': f'Sample file {safe_name} not found'}), 404

    try:
        diagnosis = predict_tumor(path)
        diagnosis['file_path'] = url_for('get_sample_file', filename=safe_name)
        diagnosis['original_filename'] = safe_name
        diagnosis['file_size_kb'] = round(os.path.getsize(path) / 1024, 1)
        return jsonify(diagnosis), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def api_predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if model is None:
        return jsonify({'error': 'Model is not loaded. Check model.h5 file.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File format not supported.'}), 400

    try:
        orig_name = secure_filename(file.filename) or 'mri_scan.jpg'
        saved_name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{orig_name}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
        file.save(save_path)

        diagnosis = predict_tumor(save_path)
        diagnosis['file_path'] = url_for('get_uploaded_file', filename=saved_name)
        diagnosis['original_filename'] = orig_name
        diagnosis['file_size_kb'] = round(os.path.getsize(save_path) / 1024, 1)
        return jsonify(diagnosis), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Inference error: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/sample-image/<filename>')
def get_sample_file(filename):
    safe_name = secure_filename(filename)
    if os.path.exists(os.path.join(BASE_DIR, safe_name)):
        return send_from_directory(BASE_DIR, safe_name)
    if os.path.exists(os.path.join(BASE_DIR, 'templates', safe_name)):
        return send_from_directory(os.path.join(BASE_DIR, 'templates'), safe_name)
    return jsonify({'error': f'Sample file {safe_name} not found'}), 404


@app.route('/<filename>')
def get_root_file(filename):
    safe_name = secure_filename(filename)
    sample_files = {'Te-gl_0015.jpg', 'Te-meTr_0001.jpg', 'Te-noTr_0004.jpg', 'Te-piTr_0003.jpg'}
    if safe_name in sample_files or (allowed_file(safe_name) and os.path.isfile(os.path.join(BASE_DIR, safe_name))):
        return send_from_directory(BASE_DIR, safe_name)
    if safe_name in sample_files and os.path.isfile(os.path.join(BASE_DIR, 'templates', safe_name)):
        return send_from_directory(os.path.join(BASE_DIR, 'templates'), safe_name)
    return jsonify({'error': f'File {safe_name} not found'}), 404


if __name__ == '__main__':
    print("=========================================================")
    print(" NeuroScan AI - MRI Diagnostic Engine Running")
    print(f" Loaded Model: {model_path_used if model else 'None'}")
    print(" Web Interface: http://127.0.0.1:5000")
    print("=========================================================")
    app.run(host='0.0.0.0', port=5000, debug=True)

    
    
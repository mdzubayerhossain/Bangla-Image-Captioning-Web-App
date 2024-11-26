import base64  # Add this import at the top of the file
from flask import Flask, request, render_template
from PIL import Image
import io
from deep_translator import GoogleTranslator  # For translation

# Your existing imports
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/caption', methods=['POST'])
def caption():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Load the image from the user upload
    img = Image.open(file.stream).convert('RGB')

    # Conditional image captioning
    text = "a photography of"
    inputs = processor(img, text, return_tensors="pt")
    out = model.generate(**inputs)
    caption_conditional = processor.decode(out[0], skip_special_tokens=True)

    # Unconditional image captioning
    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    caption_unconditional = processor.decode(out[0], skip_special_tokens=True)

    # Translate captions to Bangla
    caption_conditional_bn = GoogleTranslator(source='en', target='bn').translate(caption_conditional)
    caption_unconditional_bn = GoogleTranslator(source='en', target='bn').translate(caption_unconditional)

    # Convert the uploaded image to base64 to display on the frontend
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

    return render_template('index.html', 
                           caption_conditional=caption_conditional_bn, 
                           caption_unconditional=caption_unconditional_bn,
                           img_data=img_base64)

if __name__ == '__main__':
    app.run(debug=True)

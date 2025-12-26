import os
import cv2  
from flask import Flask, render_template, request, jsonify
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from ultralytics import YOLO
from geopy.geocoders import Nominatim

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model = YOLO('yolov8n.pt') 
geolocator = Nominatim(user_agent="dark_side_geoint")

def get_decimal_from_dms(dms, ref):
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    if ref in ['S', 'W']:
        return -(degrees + minutes + seconds)
    return degrees + minutes + seconds

def extract_gps_data(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data: return None
        gps_info = {}
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat = get_decimal_from_dms(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
            lon = get_decimal_from_dms(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
            return {"lat": lat, "lon": lon}
    except: return None
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    coords = extract_gps_data(filepath)
    address = "Локация не определена"
    if coords:
        try:
            loc = geolocator.reverse(f"{coords['lat']}, {coords['lon']}", language='ru')
            address = loc.address if loc else address
        except: pass

    results = model(filepath)
    res_plotted = results[0].plot() 
    processed_filename = "proc_" + filename
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, res_plotted) 

    detected_objects = [model.names[int(box.cls[0])] for r in results for box in r.boxes]

    return jsonify({
        "status": "success",
        "coordinates": coords,
        "address": address,
        "objects": list(set(detected_objects)),
        "image_url": f"/static/uploads/{processed_filename}" 
    })

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']): os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rahasia123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bib_detection.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)

# Model database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

class Event(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    photos = db.relationship('Photo', backref='event', lazy=True)

class Photo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    event_id = db.Column(db.Integer, db.ForeignKey('event.id'), nullable=False)
    bib_number = db.Column(db.String(20), nullable=True)

# Inisialisasi model YOLO dan OCR
yolo_model = None
reader = None

def load_models():
    global yolo_model, reader
    try:
        # Ganti dengan path model YOLO Anda
        yolo_model = YOLO('models/best.pt')
        reader = easyocr.Reader(['en'])
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_bib_number(image_path):
    try:
        # Deteksi bib menggunakan YOLO
        results = yolo_model(image_path)
        
        # Untuk menyimpan semua bib number yang terdeteksi
        all_bib_numbers = []
        
        # Proses hasil deteksi
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop gambar untuk OCR
                img = cv2.imread(image_path)
                crop = img[y1:y2, x1:x2]
                
                # OCR pada crop
                ocr_result = reader.readtext(crop)
                
                # Ambil teks dari hasil OCR
                bib_number = ""
                for detection in ocr_result:
                    bib_number += detection[1]
                
                # Hanya ambil angka
                bib_number = ''.join(filter(str.isdigit, bib_number))
                
                if bib_number:
                    all_bib_numbers.append(bib_number)
        
        # Jika ada bib number yang terdeteksi, gabungkan dengan koma
        if all_bib_numbers:
            return ", ".join(all_bib_numbers)
        
        return None
    except Exception as e:
        print(f"Error in detection: {e}")
        return None

@app.route('/')
def index():
    # Get total photo count
    total_photos = Photo.query.count()
    
    # Get total user count
    total_users = User.query.count()
    
    # Get total event count
    total_events = Event.query.count()
    
    return render_template('home.html', 
                          total_photos=total_photos,
                          total_users=total_users,
                          total_events=total_events)

@app.route('/about')
def about():
    # Get total photo count
    total_photos = Photo.query.count()
    
    # Get total user count
    total_users = User.query.count()
    
    # Get total event count
    total_events = Event.query.count()

    return render_template('about.html',
                          total_photos=total_photos,
                          total_users=total_users,
                          total_events=total_events)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['is_admin'] = user.is_admin
            
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('search_photos'))
        
        flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validasi input
        if not username or not email or not password:
            flash('Semua field harus diisi!', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Password tidak cocok!', 'danger')
            return render_template('register.html')
        
        flash('Registrasi berhasil! Silakan login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('is_admin', None)
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    
    events = Event.query.all()
    return render_template('admin_dashboard.html', events=events)

@app.route('/admin/create_event', methods=['GET', 'POST'])
def create_event():
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        event_name = request.form['event_name']
        event_date = datetime.strptime(request.form['event_date'], '%Y-%m-%d')
        
        new_event = Event(name=event_name, date=event_date)
        db.session.add(new_event)
        db.session.commit()
        
        flash('Event created successfully!')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('create_event.html')

@app.route('/admin/upload_photos/<int:event_id>', methods=['GET', 'POST'])
def upload_photos(event_id):
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    
    event = Event.query.get_or_404(event_id)
    
    if request.method == 'POST':
        if 'photos' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        files = request.files.getlist('photos')
        
        for file in files:
            if file.filename == '':
                continue
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                
                # Deteksi nomor bib
                bib_number = detect_bib_number(file_path)
                
                # Simpan ke database
                new_photo = Photo(
                    filename=unique_filename,
                    event_id=event_id,
                    bib_number=bib_number
                )
                db.session.add(new_photo)
        
        db.session.commit()
        flash('Photos uploaded and processed successfully!')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('upload_photos.html', event=event)

@app.route('/user/dashboard')
def user_dashboard():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    
    events = Event.query.all()
    return render_template('user_dashboard.html', events=events)

@app.route('/user/search', methods=['GET', 'POST'])
def search_photos():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        event_id = request.form['event_id']
        bib_number = request.form['bib_number']
        
        # Modify search to find bib_number within comma-separated list
        photos = Photo.query.filter_by(event_id=event_id).filter(
            Photo.bib_number.like(f'%{bib_number}%')
        ).all()
        
        event = Event.query.get(event_id)
        
        return render_template('search_results.html', photos=photos, event=event)
    
    events = Event.query.all()
    return render_template('search_form.html', events=events)

def initialize():
    with app.app_context():
        db.create_all()
        
        # Buat admin jika belum ada
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                password=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
        
        # Load model
        load_models()

@app.route('/admin/view_event/<int:event_id>')
def view_event(event_id):
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    
    event = Event.query.get_or_404(event_id)
    photos = Photo.query.filter_by(event_id=event_id).all()
    
    return render_template('view_event.html', event=event, photos=photos)

@app.route('/admin/edit_event/<int:event_id>', methods=['GET', 'POST'])
def edit_event(event_id):
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    
    event = Event.query.get_or_404(event_id)
    
    if request.method == 'POST':
        event_name = request.form['event_name']
        event_date = datetime.strptime(request.form['event_date'], '%Y-%m-%d')
        
        event.name = event_name
        event.date = event_date
        
        db.session.commit()
        flash('Event updated successfully!')
        return redirect(url_for('admin_dashboard'))
    
    return render_template('edit_event.html', event=event)

@app.route('/admin/delete_event/<int:event_id>')
def delete_event(event_id):
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    
    event = Event.query.get_or_404(event_id)
    
    # Delete associated photos first
    photos = Photo.query.filter_by(event_id=event_id).all()
    
    for photo in photos:
        # Delete the actual file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], photo.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from database
        db.session.delete(photo)
    
    # Delete the event
    db.session.delete(event)
    db.session.commit()
    
    flash('Event and all associated photos deleted successfully!')
    return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    initialize()
    app.run(debug=True)
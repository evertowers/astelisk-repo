from flask import Flask, render_template, Response, redirect, url_for, flash, request, jsonify
import cv2
from dotenv import load_dotenv
import psycopg2
import face_recognition
import os
from app import app
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
url = os.getenv("DATABASE_URL")
connection = psycopg2.connect(url)
#change mo yung path sa directory nung repo
app.config["IMAGE_UPLOADS"] = "D:/astelisk-repo/astelisk-repo/static/uploads"



# Load known faces from the folder
known_faces = []
known_names = []

known_faces_folder = "known_faces/"

for filename in os.listdir(known_faces_folder):
    image = face_recognition.load_image_file(known_faces_folder + filename)
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(os.path.splitext(filename)[0])

# Access the webcam and perform real-time face recognition
def detect_faces():
    video_capture = cv2.VideoCapture(0)  # Access the webcam, change to the appropriate camera index if needed

    while True:
        # Read a frame from the video stream
        ret, frame = video_capture.read()

        # Perform face detection on the frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Perform face recognition on the detected faces
        face_names = []
        for face_encoding in face_encodings:
            # Compare the face encoding with the known face encodings
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"

            # Find the best match
            if True in matches:
                matched_index = matches.index(True)
                name = known_names[matched_index]

            face_names.append(name)

        # Draw face bounding boxes and labels on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Convert the frame to a JPEG image
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
    video_capture.release()

@app.route("/")
def home():
    return render_template("home.html")
    
@app.route("/about")
def about():
    return render_template("about.html")
    
@app.route("/base")
def base():
    return render_template("base.html")

# Kiosk Pages
@app.route("/kiosk-home")
def kioskHome():
    return render_template("kiosk-home.html")

@app.route('/kiosk-login')
def kiosk_login():
    return render_template('kiosk-login.html')

@app.route('/kiosk-logout')
def kiosk_logout():
    return render_template('kiosk-logout.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Patron Pages
@app.route('/patron-landing')
def patron_landing():
    return render_template('patron-landing_page.html')

@app.route('/patron-register')
def patron_register():
    return render_template('patron-register.html')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload-image', methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            print(image)
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))  
            return redirect(request.url)
    return render_template("patron-register.html")


# @app.route('/patron-register', methods=['POST'])
# def upload_image():
# 	if 'file' not in request.files:
# 		flash('No file part')
# 		return redirect(request.url)
# 	file = request.files['file']
# 	if file.filename == '':
# 		flash('No image selected for uploading')
# 		# return redirect(request.url)
# 	# if file and allowed_file(file.filename):
# 		filename = secure_filename(file.filename)
# 		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
# 		print('upload_image filename: ' + filename)
# 		flash('Image successfully uploaded and displayed below')
# 		return render_template('upload.html', filename=filename)
# 	# else:
# 	# 	flash('Allowed image types are -> png, jpg, jpeg, gif')
# 	# 	return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=True)
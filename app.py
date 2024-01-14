import ast
import cv2
from flask import render_template, Flask, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import os
import imageio.v3 as iio
from sklearn.cluster import KMeans
import numpy as np

dir_path = 'static/img_upload'

app = Flask(__name__)

@app.route('/')
def home(): 
      return render_template('home.html')

@app.route('/upload', methods=['POST'])
def uploadFile():
      if 'gambar' in request.files:
            file = request.files['gambar']
            filename = secure_filename(file.filename)
            file.save(os.path.join(dir_path, filename))
            return redirect(url_for('menu', filename=filename))
      return render_template('home.html')

@app.route('/menu/<filename>')
def menu(filename):
      path = '../UAS/static/img_upload/'+ filename
      img = cv2.imread(path)

      # Dapatkan lebar dan tinggi citra
      height, width, _ = img.shape

      # Check if either dimension is greater than the maximum
      if height > 700 or width > 700:
            # Calculate the scaling factor to resize the image while maintaining its aspect ratio
            scale_factor = 700 / max(height, width)

            # Resize the image using the calculated scale factor
            resized_img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))

            img = resized_img

            # Dapatkan lebar dan tinggi citra
            height, width, _ = img.shape

      # Ukuran file
      size = os.path.getsize(path) / (1024 * 1024)
      size = "%.4f" % size

      # Dapatkan jumlah bit per pixel
      bits_per_pixel = img.nbytes * 8 / (width * height)

      dominant_colors = get_dominant_colors(img)

      hex_color = [bgr_to_hex(color) for color in dominant_colors]

      return render_template('informasi_gambar.html', gambar=filename, height=height, width=width, bits=bits_per_pixel, size=size, hex_color=hex_color, dominant_colors=dominant_colors)

def get_dominant_colors(image, k=7):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

#     print(dominant_colors)

    # Convert dominant colors to hex format
    return dominant_colors

def bgr_to_hex(bgr_color):
    hex_color = "#{:02x}{:02x}{:02x}".format(*bgr_color[::-1])
    return hex_color

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    bgr_color = np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)][::-1])
    return bgr_color

def display_colors(dominant_colors):
    print("Warna yang terdeteksi pada citra:")
    for i, color in enumerate(dominant_colors):
        color_name = f"Color {i + 1}"
        hex_color = bgr_to_hex(color)
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"{color_name}: BGR {color}, HEX {hex_color}, HSV {hsv_color}")

        # Menampilkan contoh warna menggunakan cv2_imshow()
        color_patch = np.zeros((50, 50, 3), dtype=np.uint8)
        color_patch[:] = color
      #   cv2_imshow(color_patch)
        
@app.route('/change', methods=['POST'])
def change():
      filename = request.form['filename']

      path = '../UAS/static/img_upload/'+ filename
      img = cv2.imread(path)
      # Memilih warna untuk diubah
      color = request.form['color']
      # lv = len(selected_color)
      # selected_color = [int(color[i:i+2], 16) for i in (0, 2, 4)]
      selected_color = hex_to_bgr(color)

      # return selected_color
      # Memasukkan kode warna HEX untuk mengubah warna
      new_hex_color = request.form['new_color']
      new_bgr_color = np.array([int(new_hex_color[i:i+2], 16) for i in (1, 3, 5)][::-1])
      # Mengganti warna dalam citra yang mirip dengan warna yang dipilih
      threshold = 10
      mask = np.all(np.abs(img - selected_color) < threshold, axis=-1)
      img[mask] = new_bgr_color
      # print("Mask Values:", mask)

      result = 'result_' + filename
      cv2.imwrite('../UAS/static/img_result/' + result, img)
      return render_template('result_save.html', filename=filename, result=result)
      # return selected_color
from flask import Flask, render_template, request
import numpy as np
import scipy.fftpack
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio_file' not in request.files:
        return "No file part", 400

    file = request.files['audio_file']
    if file.filename == '':
        return "No selected file", 400

    analysis_type = request.form.get('analysis_type')
    filepath = "temp.wav"
    file.save(filepath)

    try:
        # Load audio file
        sample_rate, data = wavfile.read(filepath)

        # Perform analysis
        if analysis_type == 'FFT':
            transformed = np.abs(scipy.fftpack.fft(data))
            plt.plot(transformed[:len(transformed) // 2])
            plt.title("FFT Analysis")
        elif analysis_type == 'DFT':
            transformed = np.abs(scipy.fftpack.dct(data))
            plt.plot(transformed[:len(transformed) // 2])
            plt.title("DFT Analysis")
        else:
            return "Unknown analysis type", 400

        # Save plot as image
        output_image = os.path.join("static", "output.png")
        plt.savefig(output_image)
        plt.close()

        return render_template('result.html', image_file="output.png")
    finally:
        # Remove temporary file
        os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)

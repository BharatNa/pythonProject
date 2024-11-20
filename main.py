from flask import Flask, render_template, request, send_file
import numpy as np
import scipy.fftpack
from scipy.signal import butter, lfilter, spectrogram
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
        data = data / np.max(np.abs(data))  # Normalize data for filters

        output_audio = None  # Placeholder for filtered audio file

        # Perform analysis
        if analysis_type == 'FFT':
            transformed = np.abs(scipy.fftpack.fft(data))
            plt.plot(transformed[:len(transformed) // 2])
            plt.title("FFT Analysis")
        elif analysis_type == 'DFT':
            transformed = np.abs(scipy.fftpack.dct(data))
            plt.plot(transformed[:len(transformed) // 2])
            plt.title("DFT Analysis")
        elif analysis_type == 'STFT':
            f, t, Sxx = spectrogram(data, sample_rate)
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.title("STFT Analysis")
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
        elif analysis_type == 'LPF':
            b, a = butter(4, 0.2, btype='low')  # Cutoff frequency = 0.2 * Nyquist
            filtered = lfilter(b, a, data)
            output_audio = "static/lpf_output.wav"
            wavfile.write(output_audio, sample_rate, (filtered * 32767).astype(np.int16))  # Save as 16-bit PCM
            plt.plot(filtered)
            plt.title("Low Pass Filter")
        elif analysis_type == 'HPF':
            b, a = butter(4, 0.2, btype='high')  # Cutoff frequency = 0.2 * Nyquist
            filtered = lfilter(b, a, data)
            plt.plot(filtered)
            plt.title("High Pass Filter")
        elif analysis_type == 'BPF':
            b, a = butter(4, [0.2, 0.5], btype='band')  # Band: 0.2–0.5 * Nyquist
            filtered = lfilter(b, a, data)
            plt.plot(filtered)
            plt.title("Band Pass Filter")
        else:
            return "Unknown analysis type", 400

        # Save plot as image
        output_image = os.path.join("static", "output.png")
        plt.savefig(output_image)
        plt.close()

        return render_template(
            'result.html',
            image_file="output.png",
            audio_file=output_audio if output_audio else None,
        )
    finally:
        # Remove temporary file
        os.remove(filepath)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

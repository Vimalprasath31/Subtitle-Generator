

import os
import threading
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import ffmpeg
from faster_whisper import WhisperModel
import pysrt

# CPU thread tuning (adjust to your physical cores)
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('MKL_NUM_THREADS', '6')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '6')


class SubtitleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Subtitle GUI — Fast CPU (faster-whisper)")
        self.video_path = None
        self.model_name = tk.StringVar(value='small')
        self.lang_code = tk.StringVar(value='auto')
        self.skip_loudnorm = tk.BooleanVar(value=False)
        self.cancel_event = threading.Event()
        self.worker = None
        self.total_duration = None  # seconds, set after model transcribe info

        # UI
        frm = ttk.Frame(root, padding=12)
        frm.grid(row=0, column=0, sticky='nsew')

        ttk.Label(frm, text='Video file:').grid(row=0, column=0, sticky='w')
        self.file_lbl = ttk.Label(frm, text='No file selected', width=60)
        self.file_lbl.grid(row=0, column=1, sticky='w')
        ttk.Button(frm, text='Browse', command=self.browse).grid(row=0, column=2, sticky='e')

        ttk.Label(frm, text='Model:').grid(row=1, column=0, sticky='w', pady=(8, 0))
        ttk.Combobox(frm, textvariable=self.model_name, values=['tiny', 'base', 'small', 'medium', 'large'], width=10).grid(
            row=1, column=1, sticky='w', pady=(8, 0))

        ttk.Label(frm, text='Language (code or auto):').grid(row=2, column=0, sticky='w', pady=(8, 0))
        ttk.Entry(frm, textvariable=self.lang_code, width=10).grid(row=2, column=1, sticky='w', pady=(8, 0))
        ttk.Label(frm, text='(e.g. ta, hi, en) or "auto"').grid(row=2, column=2, sticky='w', pady=(8, 0))

        ttk.Checkbutton(frm, text='Skip loudnorm (faster extraction, slightly lower accuracy)', variable=self.skip_loudnorm).grid(
            row=3, column=0, columnspan=3, sticky='w', pady=(8, 0))

        self.start_btn = ttk.Button(frm, text='Start', command=self.start)
        self.start_btn.grid(row=4, column=0, pady=12)
        self.cancel_btn = ttk.Button(frm, text='Cancel', command=self.cancel, state='disabled')
        self.cancel_btn.grid(row=4, column=1, pady=12, sticky='w')

        self.progress_bar = ttk.Progressbar(frm, orient='horizontal', length=600, mode='determinate')
        self.progress_bar.grid(row=5, column=0, columnspan=3, pady=(6, 6))

        self.progress = tk.Text(frm, width=80, height=16)
        self.progress.grid(row=6, column=0, columnspan=3, pady=(4, 0))
        self.progress.configure(state='disabled')

        # make window resize nicely
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

    def log(self, *parts):
        msg = ' '.join(str(p) for p in parts) + '\n'
        self.progress.configure(state='normal')
        self.progress.insert('end', msg)
        self.progress.see('end')
        self.progress.configure(state='disabled')
        self.root.update()

    def browse(self):
        path = filedialog.askopenfilename(title='Select video',
                                          filetypes=[('Video files', '*.mp4 *.mkv *.mov *.avi *.webm'), ('All files', '*.*')])
        if path:
            self.video_path = path
            self.file_lbl.config(text=path)

    def start(self):
        if not self.video_path:
            messagebox.showwarning('No file', 'Please select a video file first')
            return
        if self.worker and self.worker.is_alive():
            messagebox.showinfo('Working', 'Transcription already running')
            return

        self.cancel_event.clear()
        self.start_btn.config(state='disabled')
        self.cancel_btn.config(state='normal')
        self.progress.configure(state='normal')
        self.progress.delete('1.0', 'end')
        self.progress.configure(state='disabled')
        self.progress_bar['value'] = 0
        self.total_duration = None

        self.worker = threading.Thread(target=self.run_pipeline, daemon=True)
        self.worker.start()

    def cancel(self):
        self.log('Cancellation requested — stopping after current chunk/segment...')
        self.cancel_event.set()
        self.cancel_btn.config(state='disabled')

    def normalize_audio(self, video_path):
        tmp = tempfile.gettempdir()
        audio_file = os.path.join(tmp, f'whisper_audio_{os.getpid()}.wav')
        if self.skip_loudnorm.get():
            self.log('Skipping loudnorm (fast extraction) ->', audio_file)
            ffmpeg.input(video_path).output(audio_file, ac=1, ar=16000).run(overwrite_output=True)
            return audio_file

        self.log('Running ffmpeg loudnorm ->', audio_file)
        # loudness normalization, mono, 16 kHz
        try:
            ffmpeg.input(video_path).output(audio_file, ac=1, ar=16000,
                                            af='loudnorm=I=-16:TP=-1.5:LRA=11').run(overwrite_output=True)
        except Exception as e:
            # fallback to simple extraction if loudnorm fails
            self.log('loudnorm failed, falling back to simple extraction:', e)
            ffmpeg.input(video_path).output(audio_file, ac=1, ar=16000).run(overwrite_output=True)
        return audio_file

    def run_pipeline(self):
        try:
            model_name = self.model_name.get().strip() or 'small'
            lang = self.lang_code.get().strip()
            if not lang or lang.lower() == 'auto':
                lang = None

            base = os.path.splitext(os.path.basename(self.video_path))[0]
            output_srt = os.path.join(os.path.dirname(self.video_path), f'{base}_english_subtitles.srt')

            self.log('Extracting and (optionally) normalizing audio...')
            audio_file = self.normalize_audio(self.video_path)
            if self.cancel_event.is_set():
                self.log('Cancelled before transcription')
                return

            # Try the fastest/best compute_type first, fallback if unsupported
            preferred_compute = 'int8_float16'
            fallback_compute = 'int8'
            model = None
            try:
                self.log(f'Attempting to load model {model_name} with compute_type={preferred_compute}')
                model = WhisperModel(model_name, device='cpu', compute_type=preferred_compute)
                self.log(f'Loaded model {model_name} with compute_type={preferred_compute}')
            except Exception as e:
                self.log(f'compute_type={preferred_compute} failed: {e}; falling back to {fallback_compute}')
                model = WhisperModel(model_name, device='cpu', compute_type=fallback_compute)
                self.log(f'Loaded model {model_name} with compute_type={fallback_compute}')

            if self.cancel_event.is_set():
                self.log('Cancelled after model load')
                return

            self.log('Transcribing (task=translate) — streaming segments now')
            segments, info = model.transcribe(audio_file, beam_size=5, language=lang, task='translate')
            self.log(f'Audio duration: {info.duration}s, detected language: {info.language}')
            self.total_duration = float(info.duration) if info.duration else None

            subs = pysrt.SubRipFile()
            idx = 1
            for seg in segments:
                if self.cancel_event.is_set():
                    self.log('Cancellation noted — stopping transcription loop')
                    break
                start_time = float(seg.start)
                end_time = float(seg.end)
                text = seg.text.strip()
                self.log(f'[{start_time:.2f} --> {end_time:.2f}]', text[:120])

                # update progress bar if duration is known
                if self.total_duration:
                    percent = (end_time / self.total_duration) * 100.0
                    self.progress_bar['value'] = min(100.0, percent)
                    self.root.update_idletasks()

                sub = pysrt.SubRipItem(
                    index=idx,
                    start=pysrt.SubRipTime(seconds=start_time),
                    end=pysrt.SubRipTime(seconds=end_time),
                    text=text
                )
                subs.append(sub)
                idx += 1

            # ensure progress bar full if finished normally
            if not self.cancel_event.is_set():
                self.progress_bar['value'] = 100

            if len(subs) > 0 and not self.cancel_event.is_set():
                subs.save(output_srt, encoding='utf-8')
                self.log('✅ Subtitles saved to', output_srt)
            else:
                self.log('No subtitles saved (empty or cancelled)')

        except Exception as e:
            self.log('❌ Error during pipeline:', e)
            messagebox.showerror('Error', str(e))
        finally:
            try:
                if 'audio_file' in locals() and os.path.exists(audio_file):
                    os.remove(audio_file)
                    self.log('Temp audio removed')
            except Exception:
                pass
            self.start_btn.config(state='normal')
            self.cancel_btn.config(state='disabled')


if __name__ == '__main__':
    root = tk.Tk()
    app = SubtitleApp(root)
    root.mainloop()

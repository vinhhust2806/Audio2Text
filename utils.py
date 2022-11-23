import os
from tabnanny import verbose
from moviepy.editor import VideoFileClip
import soundfile as sf
from pydub import AudioSegment
import math
import urllib.parse as urlparse
import json

def convert_video_to_audio_moviepy(v_path, dst):
    filename = os.path.basename(os.path.splitext(v_path)[0])
    video = VideoFileClip(v_path)
    video.audio.write_audiofile(dst, fps=16000, bitrate='256k', nbytes=2) # fps=16000, bitrate='256k', nbytes=2

class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '/' + split_filename, format="wav")
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 60)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')

def map_to_array(batch):
    speech, sampling_rate = sf.read(batch["file"])
    batch["speech"] = speech
    batch["sampling_rate"] = sampling_rate
    return batch
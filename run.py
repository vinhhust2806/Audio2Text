import glob
from setup import *
from utils import *

#video_file: video.mp4
#audio_file: audio.wav
#folder contain smaller audios: /script

convert_video_to_audio_moviepy("video.mp4", "audio.wav")
split_wav = SplitWavAudioMubin('/script', 'audio.wav')
split_wav.multiple_split(min_per_split=1)

list_wav = glob.glob('/script/*.wav')
list_wav.sort()
output = ''

for audio_file in list_wav[0:len(list_wav)]:
   ds = map_to_array({"file": audio_file})
   input_values = processor(
      ds["speech"][:,1], 
      sampling_rate=ds["sampling_rate"], 
      return_tensors="pt"
   ).input_values
   logits = model(input_values).logits[0]
   pred_ids = torch.argmax(logits, dim=-1)

   beam_search_output = ngram_lm_model.decode(logits.cpu().detach().numpy(), beam_width=500)
   output = output+' ' + beam_search_output

print(output)

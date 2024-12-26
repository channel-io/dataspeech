import numpy as np

def rms_apply(sample, audio_column_name="audio"):
    audio_array = sample[audio_column_name]["array"]
    rms = np.sqrt(np.mean(audio_array**2))
    sample["rms"] = rms
    return sample




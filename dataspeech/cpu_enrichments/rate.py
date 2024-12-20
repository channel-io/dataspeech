#from g2p import make_g2p
from g2pk import G2p

#transducer = make_g2p('eng', 'eng-ipa')
transducer = G2p()

def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):
    if isinstance(batch[text_column_name], list):  
        speaking_rates = []
        phonemes_list = []
        audio_durations_list = []
        if "speech_duration" in batch:
            for text, audio_duration in zip(batch[text_column_name], batch["speech_duration"]):
                # phonemes = transducer(text).output_string
                phonemes = transducer(text)
                audio_duration = audio_duration if audio_duration != 0 else 0.01
                speaking_rate = len(phonemes) / audio_duration
                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)
                audio_durations_list.append(audio_duration)
        else:
            for text, audio in zip(batch[text_column_name], batch[audio_column_name]):
                # phonemes = transducer(text).output_string
                phonemes = transducer(text)

                sample_rate = audio["sampling_rate"]
                audio_length = len(audio["array"].squeeze()) / sample_rate
                
                speaking_rate = len(phonemes) / audio_length
                
                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)
                audio_durations_list.append(audio_length)

        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list
        batch["audio_durations"] = audio_durations_list
    else:
        # phonemes = transducer(batch[text_column_name]).output_string
        phonemes = transducer(batch[text_column_name])
        if "speech_duration" in batch:
            audio_length = batch["speech_duration"] if batch["speech_duration"] != 0 else 0.01
        else:
            sample_rate = batch[audio_column_name]["sampling_rate"]
            audio_length = len(batch[audio_column_name]["array"].squeeze()) / sample_rate

        speaking_rate = len(phonemes) / audio_length
        batch["speaking_rate"] = speaking_rate
        batch["phonemes"] = phonemes
        batch["audio_durations"] = audio_length

    return batch
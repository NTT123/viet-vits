from pathlib import Path
import unicodedata

txt_files = sorted(Path("DUMMY1").glob("*.txt"))

def clean_text(text: str):
    text = text.lower()
    text = text.replace("|", " ") # | is a special character
    text = text.replace(">", " ") # > is a special character
    text = text.replace("#", " ") # # is a special character
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.strip()
    return text

data = []
for file_path in txt_files:
    with open(file_path, "r") as f:
        text = f.read()
        text = clean_text(text)
        if len(text) > 0:
            text = ">" + text + "#"
            wav_file_path = file_path.with_suffix(".wav")
            data.append( (str(wav_file_path), text) )
        else:
            print("Skip", file_path, text)           

train_data = data[:-100]
val_data = data[-100:]       

with open("filelists/infore_audio_text_train_filelist.txt", "w") as f:
    for path, text in train_data:
        f.write(f"{path}|{text}\n")

with open("filelists/infore_audio_text_val_filelist.txt", "w") as f:
    for path, text in val_data:
        f.write(f"{path}|{text}\n")        


_, texts = zip(*data)      

alphabet = sorted(set("".join(texts)))
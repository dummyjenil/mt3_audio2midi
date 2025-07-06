```python
from huggingface_hub import hf_hub_download
from shutil import unpack_archive
from mt3_audio2midi import MT3
import nest_asyncio
nest_asyncio.apply()

unpack_archive(hf_hub_download("shethjenil/Audio2Midi_Models","mt3.zip"),"mt3_model",format="zip")
unpack_archive(hf_hub_download("shethjenil/Audio2Midi_Models","ismir2021.zip"),"ismir2021_model",format="zip")

mt3_model = MT3("mt3_model")
ismir2021_model = MT3("ismir2021_model","ismir2021")

mt3_model.predict(audio_path)
ismir2021_model.predict(audio_path)
```

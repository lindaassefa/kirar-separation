# Kirar Separation

I wanted to upload an Ethiopian song and pull the krar out by itself. That did not work.

What the site actually does is split a song into four files: `vocals.wav`, `drums.wav`, `bass.wav`, and `other.wav`. Those come from a pretrained model, HTDemucs. `other.wav` is just whatever is left after vocals, drums, and bass. It is not the krar. I left the krar download turned off so I would not pretend that stem exists.

There is also an optional mode that only makes vocals and an instrumental, using UVR-MDX-NET-Voc_FT. A lead krar can end up in the vocal file in that mode. The instrumental is the mix minus that estimate. It is still not isolated krar.

## What I failed at

The goal was a krar stem. I tried the vocal models I already had, UVR-MDX-NET-Voc_FT and Kim Inst, and listened to the results. I could still hear the drums, and the krar and the singing stayed mixed together. Spending more time tuning those two models was not getting me a krar stem, so I stopped.

I also tried a text-conditioned separator on an 8 second clip that had both krar and singing. It did not pull them apart. I never connected that experiment to the website.

I have a small classifier that can label a short clip as krar, guitar, or piano. That only names the instrument. It does not write audio, and I did not retrain it into a separator.

I also wanted a three minute song finished in under 30 seconds. On a Modal T4, the model itself took about 19 seconds for a 180 second clip. The full wait, from upload until the four wav files were back, was about 61 seconds. The container had to start from zero, and then the four files had to be sent back. So that goal was not met either. GPU inference was fast enough. The rest of the trip was not.

## Run it locally

The vocal checkpoint is not in git. Put `UVR-MDX-NET-Voc_FT.onnx` in `separator/models/` if you want the vocals-and-instrumental mode. Four stems needs the Python packages below. HTDemucs downloads its own weights the first time it runs.

```bash
python3 -m venv separator/.venv
separator/.venv/bin/pip install -r separator/requirements.txt
npm install
npm run dev
```

Open http://localhost:3000. Files can be up to 120 MB and 10 minutes.

Without Modal credentials, separation stays on the CPU. That is slow. A three minute song on my Mac was still going after several minutes, which is why I moved the same model to a T4.

The GPU path only runs if the server has Modal credentials. The browser never sees them, and they are not in this repo. If the GPU call fails, it does not silently fall back to the CPU.

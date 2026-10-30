# What I actually tried

I wanted one thing: upload a recording and get the krar by itself. I did not get that. This file is the record of the experiments, in the order I did them, including the ones that failed.

I did not train a new separator. I did not label `other` or the instrumental as krar. If a file was not really generated, the site does not offer it for download.

The machine for all of the local work was a MacBook Pro with an M2, 8 GB of RAM, 8 CPU cores, and macOS 12.5.1. That last part matters. PyTorch on this OS can see the GPU, but `torch.stft` on MPS wants macOS 14, so local spectrograms and local Demucs ran on the CPU.

## 1. The classifier was never a separator

The first model in the repo is `public/models/instrumentiq_model.onnx`. It is a small network that looks at about four seconds of audio and says krar, guitar, or piano. Class 0 is krar, class 1 is guitar, class 2 is piano. The input is one mel spectrogram, float32, shaped `[1, 1, 64, 251]`.

I matched the preprocessing to the Colab notebook that trained it, including the librosa mel settings. Against librosa, the largest absolute difference in the features was **7.451e-8**. So the browser preprocess is doing the same math. That only means the label is trustworthy. The network still cannot write a stem. I left it in the repo and did not retrain it.

## 2. Vocal removal was the first separation I could actually run

The first real audio outputs were `vocals.wav` and `instrumental.wav` from UVR-MDX-NET-Voc_FT, an ONNX MDX-Net checkpoint. It is not Demucs. The settings I used are the published ones for that checkpoint:

- sample rate 44100
- n_fft 7680
- 3072 frequency bins kept
- 256 time frames
- hop 1024
- compensate 1.021
- primary stem Vocals

One model chunk is 261120 samples. After trimming the edges, the step is 253440 samples, about 5.75 seconds. A 2:53 song is exactly 31 of those chunks. The instrumental is just the mix minus the vocal estimate. There is no second network for it.

I also tried Kim Inst, another UVR ONNX checkpoint, because Voc_FT sometimes treats a lead krar like a voice. I ran both on singing, on krar-dominated excerpts, and on excerpts that had both. I exported the original, the vocal estimate, and the instrumental estimate, and I listened. I did not use the classifier confidence, or how much energy was left, as proof that the separation was clean.

The listening test failed the actual goal. I could hear drums in the result, and the krar and the singing were still in the same files. More tuning of Kim Inst or Voc_FT was not going to become a krar separator, so I stopped treating either model as one. Voc_FT stayed in the app only as the optional "vocals and instrumental" mode. In that mode a lead krar can still land in the vocal file. The page says so.

On this Mac, the slow part of Voc_FT was ONNX Runtime on the CPU. I kept one session loaded instead of rebuilding it per song. I tried thread counts. 8 intra-op threads were the fastest, about 17.1 seconds per chunk. 4 threads were 17.9 seconds, 2 were 23.2, and 1 was 39.7. CoreML compiled only 151 of 178 nodes, ran slower, and changed the waveform, so I did not use it. One later 10 second API run took 78.4 seconds end to end and 70.1 seconds of inference, but the machine was out of memory then. I do not count that as the model getting worse.

## 3. A text-conditioned model also failed

After the vocal models failed, I tried a pretrained text-conditioned separator on one 8 second excerpt that had both krar and singing. I exported the original, an estimated krar, estimated vocals, and a residual. I did not connect it to the website until I had listened.

It did not separate them. The estimated krar still tracked the mixture, with a correlation around 0.94. I left that code unwired. I also did not download more checkpoints after that, and I did not run SAM-Audio, because this 8 GB machine cannot hold it.

## 4. What the website became

The page is a separator, not a classifier demo. You upload a file of at most 120 MB and 10 minutes. The default run writes four WAVs and lets you play or download each one:

- vocals
- drums
- bass
- other

`other` is the leftover instruments from HTDemucs. The krar card stays on the page and stays disabled, with a note that other and the instrumental are not isolated krar. If the separation did not create a file, its download returns 404.

The optional mode is still Voc_FT: vocals plus instrumental. Jobs live in a temp directory and are deleted after the previews are fetched, on failure, and if they are older than two hours. Uploads are limited to 6 every 10 minutes.

## 5. Moving the vocal model to a T4

CPU Voc_FT was too slow for a full song, so I benchmarked the same ONNX file on Modal. I approved a ceiling of $2 for one T4 run and one L4 run. The rates I was looking at were about $0.590 per T4 hour, $0.800 per L4 hour, $0.0473 per CPU-core hour, and $0.008 per GiB hour. The worker asks for 2 CPU cores and 8 GiB. An always-on T4 would be roughly $425 a month, so I did not turn that on.

The T4 benchmark ran. Two tries at the L4 died while the audio was still uploading, before a container started. I did not retry the L4. The T4 is the GPU this project uses.

I then put a private worker in front of the site, not a public website deploy. The worker is app `kirar-vocal`, class `VocalSeparator`. It is allowed one container, a 5 minute timeout, and no retries. `min_containers` is 0, so it scales to zero. After the last song it stays up for 3 minutes, which is about $0.037 at the T4 rate, and then it exits. Tokens stay on the server. If the GPU call fails or the month's bill plus $0.15 would pass the limit I set ($2, and the code refuses anything above $5), the site returns an error. It does not quietly run the long CPU job instead.

The one vocal end-to-end test was a 10 second clip.

| | |
| --- | --- |
| Upload until both WAVs were saved | 30.5 s |
| Server received the file until it was ready | 28.4 s |
| GPU inference | 9.05 s |
| Cold start after the call was accepted | 8.64 s |
| Time to accept the call | 6.12 s |
| Decode / export | 0.12 s / 0.01 s |

The provider was CUDA. Both files were 44.1 kHz, 10.000 seconds, and 1,764,044 bytes. Krar was 404. The metered cost of that test, including the warm window, was **$0.04069**. The container was gone after the idle window.

## 6. Four stems, still not krar

People expect vocals, drums, bass, and other, so I added that with the official pretrained `htdemucs` from Demucs 4.0.1. It is a bag of one Hybrid Transformer Demucs. The source names are exactly drums, bass, other, and vocals. The sample rate is 44100 and the audio is stereo. The longest segment that checkpoint can run is 7.8 seconds. I did not train it, fine-tune it, or swap it for another checkpoint.

The first proof was local, on the CPU, on the first 6 seconds of an azmari recording. One segment, because the clip is shorter than 7.8 seconds.

| | |
| --- | --- |
| Upload until the four WAVs were saved | 66.2 s |
| Decode | 1.92 s |
| Inference | 54.0 s |
| Export | 0.12 s |
| Device | CPU |

Each stem was 44.1 kHz, 16-bit stereo, exactly 6.000 seconds. Vocals and other were not the same signal (correlation 0.11). Vocals and drums were basically unrelated (correlation 0.01). Adding the stems back up matched the mix at correlation 0.9999. Drums and bass were almost silent. That was the recording, which is singing and krar, not a broken model. I played all four files. Krar and the instrumental correctly 404'd in that mode.

Then I tried a real song, 173.5 seconds, an MP3. The CPU worker reported progress as 0 of 1 for the entire `apply_model` call, so the page could not say where it was. After about 14 minutes it was still on that first step. I killed it. A linear guess from the 6 second timing is tens of minutes for a 3 minute song. I did not start another CPU run of it.

## 7. The speed experiment

The new goal was a 3 minute song, four stems, in under 30 seconds, without changing the model and without a quality bake-off. I changed how the same HTDemucs checkpoint is run.

Demucs can repeat the whole song with a random time shift and average the passes. `shifts=1` is one shifted pass. I set `shifts=0`, so there is one plain pass. The library says the shift is worth up to about 0.2 SDR. I gave that up for speed. I kept the normal 0.25 overlap, because dropping it only removes a few segments and makes the seams worse. The segment length stays at the checkpoint maximum, 7.8 seconds. At 0.25 overlap the stride is 257985 samples, so a 180 second song is 31 segments.

I run each segment under `torch.inference_mode()`, and on CUDA I turn on cuDNN benchmark because those segments are a fixed size. I did not switch the model to float16. The official inference path is float32, and I was not going to spend the one GPU test finding out that half precision broke the spectrogram. The model is loaded once when the container starts and reused. The browser sends the original file once. The worker decodes it once and returns 16-bit WAVs. Progress is the segment count, so the page can show 7 of 31 instead of sitting on 0 of 1.

The live worker before this change only had the vocal method. Deploying the new image was CPU work and did not start a GPU by itself, because `min_containers` is still 0.

## 8. The one 180 second GPU test

I cut the first 180.000 seconds of the same azmari recording and sent that file through the local site, with the server pointed at the new T4 worker. One song. No second run.

| | |
| --- | --- |
| Upload until all four WAVs were saved | **60.6 s** |
| Server received the file until the stems were ready | 56.9 s |
| GPU inference | **18.9 s** |
| Cold start after Modal accepted the call | 11.9 s |
| Time to accept the call, including sending the audio | 4.4 s |
| Decode on the worker | 0.38 s |
| Writing the WAVs on the worker | 0.72 s |
| Segments | 31 |
| Device | CUDA |
| Model | htdemucs |
| Shared gain | 0.899 |

The first segment did not finish until 30.6 seconds. That already spent the whole budget. The other 30 segments then finished in 13.2 seconds, about 0.44 seconds each. All 31 were done at 43.8 seconds. The job was not marked done until 58.6 seconds. Those last 14.8 seconds were the four WAV files coming back. Each file is 31,752,044 bytes, 44.1 kHz, 16-bit stereo, exactly 180.000 seconds.

I checked that they are real stems and not copies of each other. On a downsampled listen:

| Stem | RMS | Peak |
| --- | --- | --- |
| Vocals | 1667 | 28821 |
| Drums | 1402 | 30197 |
| Bass | 5.5 | 390 |
| Other | 546 | 12799 |

Vocals against drums correlated at 0.03. Vocals against other correlated at 0.19. Bass is quiet again because this excerpt does not have much bass, and the file is still a distinct stem, not a duplicate. Krar and instrumental returned 404.

The Modal meter went from $0.061 to $0.136, so this deploy plus the test cost **$0.075**. The container started at 17:33 and the container list was empty at 17:38, after the three minute idle window. I then took the Modal credentials back off the local server, so a later upload on localhost does not spend money.

## 9. Did I hit 30 seconds?

No.

The model pass itself, 18.9 seconds, is under 30. The thing a person waits for is not. 60.6 seconds from upload to the four files.

The biggest piece of this particular run was the cold start. Nothing finished until 30.6 seconds, because the T4 was scaled to zero and the first segment also had to warm CUDA up. After that the segments were fast.

A warm container would still miss 30 seconds. Inference was 18.9 seconds, and returning the four uncompressed WAVs took another 14.8 seconds after the last segment. Those two alone are about 34 seconds, before counting the upload. So the next limit, once the GPU is already awake, is shipping four full-length WAV files back, not the Mac CPU and not the choice of overlap.

## What I am willing to say

I can separate a song into vocals, drums, bass, and other with a pretrained HTDemucs model, and I can do the vocal-only path with UVR-MDX-NET-Voc_FT. On a T4 the four-stem model is fast enough that a 3 minute song is a few tens of seconds of GPU work. I cannot say the krar is isolated. I cannot say a 3 minute song is done in under 30 seconds, because the one time I measured the whole trip it took 60.6 seconds.

Custom Wake Word
===

Data Collection
---

Record 1-second snippets of various people saying the target word with different inflections.

Ideally, you want to use the same microphone as you would for deployment. If you plan to deploy to a single board computer, run the following command in Linux:

```
arecord --device=hw:1,0 --format S16_LE --rate 44100 -c1 test.wav -V mono
```

Say the target words a number of times (you will want at least 50 samples total). Press 'ctrl' + 'c' when you are done recording.
# Media Assets

The `media/` folder contains Manim output (videos, images, and LaTeX renders).
Reference PDFs and the written paper live under `docs/`.

Repository policy:
- All generated Manim assets under `media/` are gitignored.
- PDFs under `docs/` are kept under version control.

To regenerate animations locally:
```bash
manim -pqh mw_animation.py MultiplicativeWeightsAnimation
manim -pqh wisdom_of_crowd.py WisdomOfTheCrowd
```

If you add new reference PDFs, place them under `media/` so they remain tracked.

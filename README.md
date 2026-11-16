# Functional Safety Project
Learning how to build a human detection system through Computer Vision + IR Sensor.
Modeling the Digit 5 human safety features.

1. Validate the detection/shutdown stack
2. Test on MyCobot280
3. Add auxillary: LED, buzzer
4. Build new robot..?

More info here: 
https://docs.google.com/document/d/1pPgRoq7x-HOmyd06Knj9YyIjx2h6n3ikXu2xJOf4RG8/edit?tab=t.0#heading=h.ekrlz78icm8j

*This project is explictly for learning so here are sourced I used*
- https://docs.ultralytics.com/modes/predict

## Layout
```
functional_safety/      # the Python package - all imports are absolute from here
├── paths.py            # repo paths (models/, assets/) resolved from __file__
├── shared/             # code used by both targets
│   ├── constants.py
│   └── cv.py           # CV class + crunch/heat helpers
├── raspi/              # Raspberry Pi target (picamera2, AMG8833, gpiozero)
│   ├── hw.py
│   └── main.py
└── mac/                # Mac target: redundant camera + serial stream viewer
    ├── main.py
    └── stream.py

models/                 # YOLO weights, loaded via functional_safety.paths
assets/
notebooks/              # training.ipynb
```

## Running
Run entry points as modules from the repository root:
```
python -m functional_safety.raspi.main
python -m functional_safety.mac.main
```
Importing `functional_safety.raspi` no longer pulls in the Pi-only hardware
libraries, so the shared code can be imported on a Mac.

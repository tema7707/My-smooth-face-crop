# My smooth face shifter

This repository contains an implementation of my smooth face shifter.  This algorithm solves the problem of face shaking when transferring a face to a video. The [face-alignment](https://github.com/1adrianb/face-alignment "face-alignment") library was used to detect the key points of the face, and the [FaceShifter](https://github.com/taotaonice/FaceShifter "FaceShifter") network was used to transfer the face.

### Goals

- [x] Face detection
- [x] Cropping
    - [x] Padding
    - [x] Ratation
    - [ ] Align with [Active Appearance Model](https://www.menpo.org/menpofit/aam.html#warp)
- [x] Overlay
- [x] Face swap with FaceShifter
 - [ ] Fine-tune
- [x] Video processing
- [ ] Processing multiple faces


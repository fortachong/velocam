# TODO

## 2023-09-09
### Models
1. Car Detection. There are some OpenVino models already compiled: [https://docs.openvino.ai/2023.0/omz_models_model_vehicle_detection_adas_0002.html](https://docs.openvino.ai/2023.0/omz_models_model_vehicle_detection_adas_0002.html). Also: [https://docs.openvino.ai/2023.0/omz_models_model_vehicle_detection_0202.html](https://docs.openvino.ai/2023.0/omz_models_model_vehicle_detection_0202.html)
A model vehicle-detection-adas-0002: [https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-detection-adas-0002/model.yml](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/vehicle-detection-adas-0002/model.yml)
2. Tracking
3. Optical Flow (not sure yet)
4. We need some way to estimate speed and direction
5. OAKD provides a way to measure distance, but we need to keep a decent FPS, so all calculations have to be kept light.

### Prototype
Build a prototype using OAK-D device and a laptop on the host side. 

### Infrastructure
1. Check feasibility using Raspberry Pi.
2. Otherwise a small factor pc. Check if there is a way to power a small factor pc with small acid bateries.
3. What about display


---
## Some Useful Links

### Luxonis API Documentation
[https://docs.luxonis.com/projects/api/en/latest/](https://docs.luxonis.com/projects/api/en/latest/)


### Luxonis Community Discord Channel
[https://discord.com/channels/790680891252932659/924798753330831370](https://discord.com/channels/790680891252932659/924798753330831370)

### Some videos on Youtube

[https://www.youtube.com/watch?v=bcFwnXN_Sx0](https://www.youtube.com/watch?v=bcFwnXN_Sx0)

[https://www.youtube.com/watch?v=IBeh30PQOYI](https://www.youtube.com/watch?v=IBeh30PQOYI)

[https://www.youtube.com/watch?v=e_uPEE_zlDo](https://www.youtube.com/watch?v=e_uPEE_zlDo)

[https://www.youtube.com/watch?v=7BkHcJu57Cg](https://www.youtube.com/watch?v=7BkHcJu57Cg)

[https://www.youtube.com/watch?v=HHJDXE0VcAY](https://www.youtube.com/watch?v=HHJDXE0VcAY)

### Openvino model zoo github repository
[https://github.com/openvinotoolkit/open_model_zoo/tree/master](https://github.com/openvinotoolkit/open_model_zoo/tree/master)

### PINTO model zoo github repository
[https://github.com/PINTO0309/PINTO_model_zoo/tree/main](https://github.com/PINTO0309/PINTO_model_zoo/tree/main)

### YOLO on device decoding
[https://github.com/luxonis/depthai-experiments/tree/master/gen2-yolo/device-decoding](https://github.com/luxonis/depthai-experiments/tree/master/gen2-yolo/device-decoding)

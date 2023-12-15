

# inference
Copy v8_inference.py to ultralytics root directory, and execute.

```
$ rm -rf ./david/detect/inference/result/*;python3 v8_inference.py --input=./david/detect/inference/test.mp4 --model_file=/home/david/code/yolo/ultralytics/runs/detect/train2/weights/best.pt --output_dir=./david/detect/inference/result --interval=0 --save_video
```


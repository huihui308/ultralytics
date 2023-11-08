
# Construct envirment
```
$ conda create -n V8 python=3.8.16
$ pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```



# Track remote fork warehouse

```
$ git remote add upstream https://github.com/ultralytics/yolov5.git
$ git remote -v
$ git fetch upstream
$ git checkout main
$ git merge upstream/main
$ git checkout david
$ git merge main
```


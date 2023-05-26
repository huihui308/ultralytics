

# Make docker image
To begin with, put Arial.ttf and Arial.Unicode.ttf to current directory.

Make image:
```
$ docker build -t davidv8:v1 .
```

通过docker ps -a查看容器ID，例如：f4f9d52e741d
```
$ docker ps -a
```

# Run docker
Install nvidia-container-toolkit on machine.
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install nvidia-container-toolkit
service docker restart
```


docker run -t -i -v /home/david/dataset:/soft 容器id /bin/bash
```
$ docker images
REPOSITORY   TAG       IMAGE ID       CREATED             SIZE
davidv8      v1        2ddbed870f64   About an hour ago   7.65GB
$ docker run -t -i --gpus "device=0,1" -v /home/david/dataset:/home/david/dataset 2ddbed870f64 bash
or all gpu
$ docker run -t -i --gpus all -v /home/david/dataset:/home/david/dataset 8edad533cbf6 bash

$ docker run -itd --gpus all --name davidv8 -p 5005:22 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -v /home/david/dataset:/home/david/dataset --ipc=host 2ddbed870f64
```


使用-v参数来挂载主机下的一个目录，把/opt文件挂载在这个容器上做共享文件夹，这样启动后容器会自动在根目录下创建soft文件夹，同时也就要求了**/soft必须写绝对路径**

进入镜像docker exec -it 容器ID /bin/bash
退出 ：ctrl+D
启动容器 docker start 容器id
停止容器 docker stop 容器id
重启容器 odcker rstart 容器id
删除容器 docker rm -f 容器id


Backup docker image
```
$ docker ps -a
CONTAINER ID   IMAGE          COMMAND   CREATED              STATUS                     PORTS     NAMES
84c796788086   2ddbed870f64   "bash"    About a minute ago   Exited (0) 4 seconds ago             ecstatic_morse
$ docker stop 84c796788086
$ docker commit -p 84c796788086 davidv8-init
sha256:8edad533cbf6812f2e8ed3f935ab30ed781aee03a6b7ae7bc1e744354bedf729
$ docker save -o ./davidv8-init.tar davidv8-init

```

Load docker image:
```
$ docker load -i ./davidv8-init.tar
```





# Add docker command to user group
Docker daemon 绑定的是 Unix socket，这就导致 docker 需要 root 权限才能使用，但这十分麻烦，因为其他用户必须经常使用 sudo。

为此，docker daemon 创建 Unix socket 时，会允许所有 docker 组内成员访问，所以我们只需要将用户加入 docker 组内就可以避免使用 sudo。

* 创建 docker 组：sudo groupadd docker
* 将用户加入 docker 组内：sudo usermod -aG docker $USER
* 重新登录

# Reference
&ensp;&ensp;https://blog.csdn.net/GODLwc/article/details/128969083<br/>
&ensp;&ensp;https://zinglix.xyz/2021/04/03/docker-group/<br/>
&ensp;&ensp;https://www.jianshu.com/p/6eea18d6fb39<br/>
&ensp;&ensp;https://www.cnblogs.com/linhaifeng/p/16108285.html<br/>
&ensp;&ensp;<br/>
&ensp;&ensp;<br/>
&ensp;&ensp;<br/>
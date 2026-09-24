sudo docker run -itd \
  --name sglangmain_npu_cryang \
  --shm-size=512g \
  --privileged \
  --net=host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /etc/localtime:/etc/localtime \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /home/youhuibai/cryang:/home/cryang \
  -v /mnt/raid/user/data:/mnt/raid/user/data \
  quay.io/ascend/sglang:main-cann9.0.0-a3 \
  /bin/bash



sudo docker run -itd \
--name sglang0520_npu_cryang \
--shm-size=512g \
--privileged \
--net=host \
--ulimit memlock=-1:-1 \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /etc/hccn.conf:/etc/hccn.conf \
-v /etc/localtime:/etc/localtime \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /dev/infiniband:/dev/infiniband \
-v /sys/class/infiniband:/sys/class/infiniband:ro \
-v /sys/class/net:/sys/class/net:ro \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /home/youhuibai/cryang:/home/cryang \
-v /mnt/raid/user/data:/mnt/raid/user/data \
-v /data_lib/data/models:/data_lib/data/models \
quay.io/ascend/sglang:v0.5.20-cann9.0.0-a3 \
/bin/bash
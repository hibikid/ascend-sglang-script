sudo docker run -itd \
  --name sglang0516_npu_cryang \
  --shm-size=512g \
  --privileged \
  --net=host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /etc/hccn.conf:/etc/hccn.conf \
  -v /etc/localtime:/etc/localtime \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
  -v /mnt/raid/user/data:/mnt/raid/user/data \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /home/youhuibai/cryang:/home/cryang \
  quay.io/ascend/sglang:v0.5.16-cann9.0.0-a3 \
  /bin/bash
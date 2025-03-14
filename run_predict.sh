#!/bin/bash
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export MIOPEN_DEBUG_CONV_DIRECT=0
export MIOPEN_DEBUG_CONV_WINOGRAD=0
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
export NCCL_SOCKET_IFNAME=ib0
export HSA_USERPTR_FOR_PAGED_MEM=0
# export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5_0

lrank=$OMPI_COMM_WORLD_LOCAL_RANK
comm_rank=$OMPI_COMM_WORLD_RANK
comm_size=$OMPI_COMM_WORLD_SIZE

APP="python `pwd`/predict.py --master_ip ${1} --world_size ${comm_size} --rank ${comm_rank}"

case ${lrank} in
[0])
  export HIP_VISIBLE_DEVICES=0
  export UCX_NET_DEVICES=mlx5_0:1
  export UCX_IB_PCI_BW=mlx5_0:50Gbs
  echo NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=0 --membind=0 ${APP}
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=0 --membind=0 ${APP}
  ;;
[1])
  export HIP_VISIBLE_DEVICES=1
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  echo NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=1 --membind=1 ${APP}
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=1 --membind=1 ${APP}
  ;;
[2])
  export HIP_VISIBLE_DEVICES=2
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  echo NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=2 --membind=2 ${APP}
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=2 --membind=2 ${APP}
  ;;
[3])
  export HIP_VISIBLE_DEVICES=3
  export UCX_NET_DEVICES=mlx5_1:1
  export UCX_IB_PCI_BW=mlx5_1:50Gbs
  echo NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=3 --membind=3 ${APP}
  NCCL_SOCKET_IFNAME=ib0 numactl --cpunodebind=3 --membind=3 ${APP}
  ;;
esac

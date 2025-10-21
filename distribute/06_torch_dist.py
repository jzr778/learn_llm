import os 
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def init_process(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

# def run(rank, size):
#     time.sleep(1)
#     print(rank, size)


# gloo，走cpu
# def run(rank, size):
#     tensor = torch.zeros(1)
#     # print("tensor:",tensor)
#     if rank == 0:
#         tensor += 1
#         dist.send(tensor=tensor, dst=1)
#     else:
#         dist.recv(tensor=tensor, src=0)
#     print('Rank ', rank, ' has data ', tensor[0])

# nccl，走gpu
# def run(rank, size):
#     tensor = torch.zeros(1).to(rank)
#     print("tensor:",tensor)
#     if rank == 0:
#         tensor += 1
#         dist.send(tensor=tensor, dst=1)
#     if rank == 1:
#         print("init tensor", tensor)
#         dist.recv(tensor=tensor, src=0)
#     print('Rank ', rank, ' has data ', tensor[0])


def run(rank, size):
    group = dist.new_group([0,1])
    if rank == 0:
        tensor = torch.tensor([1.,2.,3.])
    else:
        tensor = torch.tensor([4.,5.,6.])
    tensor = tensor.to(rank)
    print(f'Rank: {rank}, random tensor: {tensor}')
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f'Rank: {rank}, has data: {tensor}')

if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank,size,run))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("finished")
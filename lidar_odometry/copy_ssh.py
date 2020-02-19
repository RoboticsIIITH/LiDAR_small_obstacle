# from paramiko import SSHClient
# from scp import SCPClient
import os

# ssh = SSHClient()
# ssh.load_system_host_keys()
# ssh.connect('udit@10.4.24.24')

# SCPCLient takes a paramiko transport as an argument
# scp = SCPClient(ssh.get_transport())

root_dir = "/scratch/ash/IIIT_Labels/train/"
dest_dir = "/scratch/ash/IIIT_Labels/train/"

# folders = ["image","labels","context_full","context_temporal_template"]
folders = ["labels"]
file_name = open("copy_paths.txt","w+")

seq_names = os.listdir(root_dir)
for seq in seq_names:
    for folder in folders:
        path = os.path.join(root_dir,seq,folder)
        # dest_path = os.path.join(dest_dir,seq,folder)
        # scp.put(path, recursive=True, remote_path=dest_path)
        file_name.writelines(path)
# scp.close()
file_name.close()

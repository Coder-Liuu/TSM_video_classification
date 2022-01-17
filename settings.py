# MODEL
framework = 'Recognizer2D'

# backbone 
pretrained = 'TSM_k400.pdparams'
num_seg = 8
layers = 50

# head
num_classes = 2
in_channels = 2048
drop_ratio = 0.8
std = 0.001

# DATASET
batch_size = 4
train_file_path = 'annotation/violence_train_videos.txt'
valid_file_path = 'annotation/violence_val_videos.txt'
suffix = ''
train_shuffle = True
valid_shuffle = False
return_list = True

# OPTIMIZER
momentum = 0.9
boundaries = [10, 20]
values = [0.001, 0.0001, 0.00001]
clip_norm = 20.0

model_name = 'TSM'
log_interval = 10
save_interval = 1
epochs = 10

# device
device = 'gpu'

# Color
Color = {
    'RED': '\033[31m',
    'HEADER': '\033[35m',  # deep purple
    'PURPLE': '\033[95m',  # purple
    'OKBLUE': '\033[94m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m'
}


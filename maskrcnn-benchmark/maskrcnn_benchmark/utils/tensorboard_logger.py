from tensorboardX import SummaryWriter


class TensorboardXLogger(object):
    def __init__(self, log_dir='../output/tensorboard', is_training=True):
        self.writer = SummaryWriter(log_dir)
        if is_training:
            self.phase = 'train'
        else:
            self.phase = 'valid'


    def write(self, meters, iteration):
        for name, meter in meters.meters.items():
            if name.split('_')[0] != 'loss':
                continue
            name = name + '_' + self.phase
            self.writer.add_scalars(name, {'median': meter.median, \
                                           'global_average':meter.global_avg}, iteration)

    def export_to_json(self, filename="./all_scalars.json"):
        self.writer.export_scalars_to_json(filename)
        self.writer.close()
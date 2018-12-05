from tensorboardX import SummaryWriter


class TensorboardXLogger(object):
    def __init__(self, log_dir='../output/tensorboard'):
        self.writer = SummaryWriter(log_dir)

    def write(self, meters, iteration, phase='Train'):
        for name, meter in meters.meters.items():
            if name.split('_')[0] != 'loss' and name.split('_')[0] != 'lr' and name != "total_loss":
                continue
            name = name + '_' + phase
            # if phase == 'Train':
            #     self.writer.add_scalars(name, {'median': meter.median, \
            #                                'global_average':meter.global_avg}, iteration)
            # else:
            #     self.writer.add_scalar(name,  meter.global_avg, iteration)
            self.writer.add_scalar(name, meter.median, iteration)

    def export_to_json(self, filename="./all_scalars.json"):
        self.writer.export_scalars_to_json(filename)
        self.writer.close()
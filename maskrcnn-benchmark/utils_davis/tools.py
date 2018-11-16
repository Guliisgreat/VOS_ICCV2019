import time
import os

class TimerBlock:
    def __init__(self, title, logger):
        self.logger = logger
        logger.info(("{}".format(title)))

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.clock()
        self.interval = self.end - self.start

        if exc_type is not None:
            self.log("Operation failed\n")
        else:
            self.log("Operation finished\n")

    def log(self, string):
        duration = time.clock() - self.start
        units = 's'
        if duration > 60:
            duration = duration / 60.
            units = 'm'
        self.logger.info(("  [{:.3f}{}] {}".format(duration, units, string)))

    @staticmethod
    def log2file(fid, string):
        fid = open(fid, 'a')
        fid.write("%s\n" % string)
        fid.close()


def get_exp_output_dir(exp_name, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    exp_output_dir = os.path.join(output_dir, exp_name)
    if not os.path.exists(exp_output_dir):
        os.mkdir(exp_output_dir)
    return exp_output_dir

from optparse import OptionParser
import sys
from model.hyperparameters import Hyperparameters
import model.training as training

def parseArgs(argsv):
    usage = """
    USAGE:      python smbot.py <options>
    Examples:   python smbot.py --mode demo
    """
    parser = OptionParser(usage)
    parser.add_option('-m', '--mode', dest='mode', 
                      help='The MODE to execute: demo, train', metavar='MODE', default='demo')
    parser.add_option('-n', '--name', dest='name', 
                      help='The NAME of the model to demo', metavar='NAME', default='agent')
    parser.add_option('--world', dest='world', type=int,
                      help='The demo WORLD to play', metavar='WORLD', default=1)
    parser.add_option('--stage', dest='stage', type=int,
                      help='The demo STAGE to play', metavar='STAGE', default=1)
    parser.add_option('--demoscale', dest='demo_scale', type=int,
                      help='The SCALE to multiply the NES demo viewer resolution by', metavar='STAGE', default=4)

    options, _ = parser.parse_args(argsv)
    return options

if __name__ == '__main__':
    options = parseArgs(sys.argv[1:])
    if options.mode == 'demo':
        training.demo(options.name, options.world, options.stage, options.demo_scale, Hyperparameters())
    if options.mode == 'train':
        training.trainModel(options.name, Hyperparameters())
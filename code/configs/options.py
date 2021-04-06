import argparse
import os

class Options():
    def __init__(self, parser):
        self.initialized = False

    def initialize(self, parser):
        
        # Dataset options
        parser.add_argument('--exp', type=str, default='exp/01',         help='name of the exp folder')
        parser.add_argument('--profile', type=str, default='240',        help='name of the cluster profile')
        self.initialized = True
        return parser
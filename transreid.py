import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRANSREID_DIR = os.path.join(_THIS_DIR, 'TransReID')
if _TRANSREID_DIR not in sys.path:
    sys.path.insert(0, _TRANSREID_DIR)

from config import cfg
from model import make_model
from utils.logger import setup_logger

# cfg_path = os.path.join(_TRANSREID_DIR, 'configs', 'market.yml')
# cfg_path = os.path.join(_TRANSREID_DIR, 'configs', 'msmt.yml')
# cfg_path = os.path.join(_TRANSREID_DIR, 'configs', 'duke.yml')
cfg_path = os.path.join(_TRANSREID_DIR, 'configs', 'occ_duke.yml')
cfg.merge_from_file(cfg_path)
cfg.freeze()

logger = setup_logger("transreid", "", if_train=False)

if cfg_path != "":
    logger.info("Loaded configuration file {}".format(cfg_path))
    with open(cfg_path, 'r') as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
logger.info("Running with config:\n{}".format(cfg))

num_classes, camera_num, view_num = 1, 1, 1 

model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
model.load_param(cfg.TEST.WEIGHT)

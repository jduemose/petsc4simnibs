import logging

logger = logging.getLogger('PETSc4SimNIBS')
sh = logging.StreamHandler()
formatter = logging.Formatter('[ %(name)s ] %(levelname)s: %(message)s')
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)
logger.addHandler(sh)
logger.setLevel(logging.DEBUG)
logging.addLevelName(25, 'SUMMARY')

## define config

class Config(object):
    """
    default config
    """
    SAMPLE_VARIABLE = 'sample default'


class DevConfig(Config):
    """
    dev config
    """
    DEV_SAMPLE = 'sample dev'


class DeployConfig(Config):
    """
    deployment config
    """
    DEPLOY_SAMPLE = 'sample deploy'

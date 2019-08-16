class GenericAgent:
    """Super class for a generic agent, ensures that some properties are set when instantiating the agent"""

    def __init__(self, config):
        self.check_required_properties(config, self.required_properties())

    def check_required_properties(self, config: dict, required_properties: list):
        for property in required_properties:
            assert config.keys().__contains__(property), f"'{property}' is required in the configuration dictionary"

    def required_properties(self):
        return []

    def update_from_config(self, config: dict):
        for key in config.keys():
            setattr(self, key, config[key])

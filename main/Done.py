class Done():
    def __init__(self, value, print_dict: dict = None, attributes: dict = None, inherit_from = None):
        assert isinstance(print_dict, dict) or print_dict == None, "print_dict must be a dictionary"
        assert isinstance(attributes, dict) or attributes == None, "attributes must be a dictionary"
        assert attributes == None or "value" not in attributes.keys(), "attributes must be not include key \"value\""
        assert isinstance(inherit_from, Done) or inherit_from == None, "Must inherit from a done object"
        
        self.value = value
        self.print_dict = print_dict if print_dict != None else {}
        self.attributes = attributes if attributes != None else {}

        if inherit_from != None and inherit_from.print_dict != None:
            for person in inherit_from.print_dict:
                if person not in self.print_dict:
                    self.print_dict[person] = {}
                for level in inherit_from.print_dict[person]:
                    self.print_dict[person][level] = inherit_from.print_dict[person][level]

        if inherit_from != None:
            for attr in inherit_from.attributes:
                self.attributes[attr] = inherit_from.attributes[attr]

        if attributes != None:
            for key in attributes:
                setattr(self, key, attributes[key])
import json
import argparse


class AcronymExpansionModel:
    def __init__(self, acronym_long_dict, acronym_short_dict=None, acronym_vn_dict=None, model=None):
        self.acn_dict = self.load_dict(acronym_long_dict, acronym_short_dict, acronym_vn_dict)
        self.model = model

    def expand_acronym(self, text):
        def get_acr(text):
            if text[-1] == ' ':
                return None
            return text.split(' ')[-1]
        
        acr = get_acr(text)
        if acr is None:
            return []

        full_texts = self.select(acr)
        return full_texts
    
    def load_dict(self, acronym_long_dict, acronym_short_dict, acronym_vn_dict):
        """
        MODIFY this
        
        load and preprocess the acronym dict
        """
        with open(acronym_long_dict, 'r', encoding='utf8') as f:
            acronym_dict = json.load(f)
        if acronym_short_dict is not None:
            with open(acronym_short_dict, 'r', encoding='utf8') as f:
                data = json.load(f)
                for key, value in data.items():
                    try: acronym_dict[key].extend(value)
                    except: acronym_dict[key] = value
        if acronym_vn_dict is not None:
            with open(acronym_vn_dict, 'r', encoding='utf8') as f:
                data = json.load(f)
                for key, value in data.items():
                    try: acronym_dict[key].extend(value)
                    except: acronym_dict[key] = value
        return acronym_dict

    def select(self, acronym):
        """
        MODIFY this
        
        select the full phrase from 
        an acronym in a list of options
        """
        if self.model is None:
            for key, value in self.acn_dict.items():
                if len(value) > 5:
                    self.acn_dict[key] = value[:5]
            return self.acn_dict.get(acronym, '')
        else:
            return self.model(acronym) # Model Acronym Disambiguation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Input text for model acronym disambiguation")
    parser.add_argument("--acr_long_dict", type=str, default="./result_cxr/final_long_dict.json", help="Acronym dictionary")
    parser.add_argument("--acr_short_dict", type=str, default="./result_cxr/filtered_short_dict.json", help="Acronym dictionary")
    parser.add_argument("--acr_vn_dict", type=str, default="./result_vn/all_dict.json", help="Acronym dictionary")
    parser.add_argument("--model", type=str, default=None, help="Model for acronym disambiguation")

    args = parser.parse_args()

    AcronymExpansion = AcronymExpansionModel(args.acr_long_dict, args.acr_short_dict, args.acr_vn_dict, args.model)
    
    expansion_acr = AcronymExpansion.expand_acronym(args.text)
    print(expansion_acr)



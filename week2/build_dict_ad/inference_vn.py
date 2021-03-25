import json
import argparse


class AcronymExpansionModel:
    def __init__(self, acronym_long_dict, acronym_short_dict=None, model=None):
        self.acn_dict = self.load_dict(acronym_long_dict, acronym_short_dict)
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
    
    def load_dict(self, acronym_long_dict, acronym_short_dict):
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
        return acronym_dict

    def select(self, acronym):
        """
        MODIFY this
        
        select the full phrase from 
        an acronym in a list of options
        """
        if self.model is None:
            if self.acn_dict.get(acronym, ""):
                return self.acn_dict.get(acronym)[:min(5, len(self.acn_dict.get(acronym)))]
            else: return ""
        else:
            return self.model(acronym) # Model Acronym Disambiguation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Input text for model acronym disambiguation")
    parser.add_argument("--acr_long_dict", type=str, default="./result_vn_2/final_long_dict.json", help="Acronym dictionary")
    parser.add_argument("--acr_short_dict", type=str, default="./result_vn_2/short_dict.json", help="Acronym dictionary")
    parser.add_argument("--model", type=str, default=None, help="Model for acronym disambiguation")

    args = parser.parse_args()

    AcronymExpansion = AcronymExpansionModel(args.acr_long_dict, args.acr_short_dict, args.model)
    
    expansion_acr = AcronymExpansion.expand_acronym(args.text)
    print(expansion_acr)



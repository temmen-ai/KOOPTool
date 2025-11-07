#Versie 0.9, 07-11-2024
#Versie voor pilot bij Daadkracht
import torch
import torch.nn as nn
from transformers import BertModel

class BERTFineTunerWithFeatures(nn.Module):
    def __init__(self, hidden_size, num_labels, tokenizer):
        super(BERTFineTunerWithFeatures, self).__init__()
        
        # Initialiseer het BERT-model
        self.bert = BertModel.from_pretrained('GroNLP/bert-base-dutch-cased')
        self.dropout = nn.Dropout(0.3)
        
        # Haal het [SEP] token ID direct uit de tokenizer
        self.sep_token_id = tokenizer.sep_token_id
        
        # De totale input size van de classifier is hidden_size + 6, omdat we 6 extra features hebben
        self.classifier = nn.Linear(hidden_size + 6, num_labels)

    def forward(self, input_ids, attention_mask=None, bold_features=None, indented_features=None,
                italic_features=None, underlined_features=None, numbered_features=None, volgnummer_features=None):
        # BERT model output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Identificeer de posities van de [SEP] tokens
        sep_token_indices = (input_ids == self.sep_token_id).nonzero(as_tuple=True)

        # Verzamel de hidden states van alle [SEP] tokens en combineer ze met de extra features
        logits_list = []
        for i in range(input_ids.size(0)):  # Itereer over elke sample in de batch
            sep_indices = sep_token_indices[1][sep_token_indices[0] == i]
            sep_hidden_states = last_hidden_state[i, sep_indices, :]  # Hidden states voor de [SEP]-tokens
            
            # Zorg ervoor dat de extra features dezelfde lengte hebben als de aantal [SEP]-tokens
            num_seps = len(sep_indices)

            # Selecteer de corresponderende features voor de huidige sample
            selected_bold_features = bold_features[i][:num_seps].unsqueeze(1)
            selected_indented_features = indented_features[i][:num_seps].unsqueeze(1)
            selected_italic_features = italic_features[i][:num_seps].unsqueeze(1)
            selected_underlined_features = underlined_features[i][:num_seps].unsqueeze(1)
            selected_numbered_features = numbered_features[i][:num_seps].unsqueeze(1)
            selected_volgnummer_features = volgnummer_features[i][:num_seps].unsqueeze(1)

            # Combineer de hidden states met de geselecteerde features
            combined_features = torch.cat((
                sep_hidden_states,  # Hidden states van BERT
                selected_bold_features,
                selected_indented_features,
                selected_italic_features,
                selected_underlined_features,
                selected_numbered_features,
                selected_volgnummer_features
            ), dim=1)
            
            combined_features = self.dropout(combined_features)  # Pas dropout toe

            # Voer de gecombineerde features door de classifier
            logits = self.classifier(combined_features)
            logits_list.append(logits)

        return logits_list  # Lijst van logits voor elke sample

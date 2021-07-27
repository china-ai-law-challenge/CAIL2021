import torch
from torch.nn import CrossEntropyLoss, BCELoss
from torch import nn


class MultiSpanQA(nn.Module):
    def __init__(self, pretrain_model):
        super(MultiSpanQA, self).__init__()
        self.pretrain_model = pretrain_model
        # represent start logits and end logits respectively
        self.qa_outputs = nn.Linear(pretrain_model.config.hidden_size, 2)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_labels=None,  # size: (batch_size, max_seq_length, 1)
            end_labels=None,
    ):
        outputs = self.pretrain_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_labels is not None and end_labels is not None:
            loss_fct = BCELoss(reduction="mean")
            start_loss = loss_fct(torch.sigmoid(start_logits), start_labels)
            end_loss = loss_fct(torch.sigmoid(end_logits), end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs


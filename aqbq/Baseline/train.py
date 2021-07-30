import torch
from torch.utils.data import DataLoader
from dataset import CaseData
from model import CaseClassification
from transformers import BertTokenizer, AdamW


if __name__ == "__main__":
    # check the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # prepare training  data
    training_data = CaseData('./data/train.json', class_num=252)             # 252 is the number of level3 labels
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)

    # load the model and tokenizer
    model = CaseClassification(class_num=252).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # prepare the optimizer and corresponding hyper-parameters
    epochs = 10
    learning_rate = 1e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # start training process
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(train_dataloader):
            fact, label = data

            # tokenize the data text
            inputs = tokenizer(fact, max_length=512, padding=True, truncation=True, return_tensors='pt')

            # move data to device
            input_ids = inputs['input_ids'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            label = label.to(device)

            # forward and backward propagations
            loss, logits = model(input_ids, attention_mask, token_type_ids, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 50 == 49:
                print('epoch%d, step%5d, loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    torch.save(model.state_dict(), './saved/model'+str(epochs)+'.pth')

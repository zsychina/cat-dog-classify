import torch
import torch.nn as nn
from loader import train_loader, test_loader
from lenet import LeNet
from vgg16 import VGG16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using '+str(device)+"!")

model = LeNet()
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
learn_rate = 1e-2
optimzier = torch.optim.SGD(model.parameters(), lr=learn_rate)


def train(train_loader, model, loss_fn, optimizer):
    size = len(train_loader.dataset)
    num_batches = len(train_loader)

    train_loss, train_acc = 0, 0

    for X, y in train_loader: # X是一个batch的数据，y是一个batch的标签 
        X, y = X.to(device), y.to(device)
        
        # print(len(X[0][0])) # 224

        pred = model(X)
        loss = loss_fn(pred, y)

        optimzier.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()
    
    train_acc /= size
    train_loss /= num_batches

    return train_acc, train_loss


def test(test_loader, model, loss_fn):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, test_acc = 0, 0

    with torch.no_grad():
        for imgs, target in test_loader:
            imgs, target = imgs.to(device), target.to(device)

            target_pred = model(imgs)
            loss = loss_fn(target_pred, target)

            test_acc += (target_pred.argmax(1) == target).type(torch.float).sum().item()
            test_loss += loss.item()            
    
    test_acc /= size
    test_loss /= num_batches

    return test_acc, test_loss





if __name__ == '__main__':
    # print(len(train_loader.dataset))
    # print(len(train_loader))

    epochs     = 10
    train_loss = []
    train_acc  = []
    test_loss  = []
    test_acc   = []

    # weights = torch.load('./target/finish_new.pt')
    # model.load_state_dict(weights)

    for epoch in range(epochs):
        model.train()
        epoch_train_acc, epoch_train_loss = train(train_loader, model, loss_fn, optimzier)
        
        model.eval()
        epoch_test_acc, epoch_test_loss = test(test_loader, model, loss_fn)
        
        train_acc.append(epoch_train_acc)
        train_loss.append(epoch_train_loss)
        test_acc.append(epoch_test_acc)
        test_loss.append(epoch_test_loss)
        
        template = ('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%，Test_loss:{:.3f}')
        print(template.format(epoch+1, epoch_train_acc*100, epoch_train_loss, epoch_test_acc*100, epoch_test_loss))
        
        PATH = './target/temp_new.pt'
        torch.save(model.state_dict(), PATH)
    
    print('Done')
    PATH = './target/finish_new.pt'
    torch.save(model.state_dict(), PATH)


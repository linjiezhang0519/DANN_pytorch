import torch
from torch import nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader
from itertools import cycle
from makemoons import make_trans_moons


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter))
                    - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1

class AdversarialNet(nn.Module):
    def __init__(self, in_feature, feature_size=[10], label_size=[10],
                 domain_size = [10],
                 max_iter=10000.0,
                 trade_off_adversarial='Step', lam_adversarial=1.0):
        super(AdversarialNet, self).__init__()
        #feature extractor
        feature_layers = []
        input_dim = in_feature
        for hidden_dim in feature_size:
            feature_layers.append(nn.Linear(input_dim, hidden_dim))
            feature_layers.append(nn.ReLU(inplace = True))
            input_dim = hidden_dim
        self.features = nn.Sequential(*feature_layers)

        #label_classifier
        label_layers = []
        label_imput_dim = feature_size[-1]
        if len(label_size)>0:
            for hidden_dim in label_size:
                label_layers.append(nn.Linear(label_imput_dim, hidden_dim))
                label_layers.append(nn.ReLU(inplace=True))
                label_imput_dim = hidden_dim
        label_layers.append(nn.Linear(label_imput_dim, 1))
        label_layers.append(nn.Sigmoid())
        self.label_classifier = nn.Sequential(*label_layers)

        # domain_classifier
        domain_layers = []
        domain_imput_dim = feature_size[-1]
        if len(domain_size)>0:
            for hidden_dim in domain_size:
                domain_layers.append(nn.Linear(domain_imput_dim, hidden_dim))
                domain_layers.append(nn.ReLU(inplace=True))
                domain_imput_dim = hidden_dim
        domain_layers.append(nn.Linear(domain_imput_dim,1))
        domain_layers.append((nn.Sigmoid()))
        self.domain_classifier = nn.Sequential(*domain_layers)

        # parameters
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter
        self.trade_off_adversarial = trade_off_adversarial
        self.lam_adversarial = lam_adversarial
        self.__in_features = 1

    def forward(self, x):
        if self.training:#记录训练次数
            self.iter_num += 1
        if self.trade_off_adversarial == 'Cons':
            coeff = self.lam_adversarial
        elif self.trade_off_adversarial == 'Step':
            coeff = calc_coeff(self.iter_num, self.high,
                               self.low, self.alpha, self.max_iter)
        else:
            raise Exception("loss not implement")
        x = x * 1.0
        x.requires_grad_(True)
        x = self.features(x)
        label_predict = self.label_classifier(x)

        x = x.clone().detach().requires_grad_(True)#克隆梯度
        x.register_hook(grl_hook(coeff))
        domain_predict = self.domain_classifier(x)
        return label_predict, domain_predict

    def output_num(self):
        return self.__in_features

class EarlyStopping():

    def __init__(self, patience = 20, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.count =0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss-val_loss > self.min_delta:
            self.best_loss = val_loss
            self.count = 0
        elif self.best_loss -val_loss <self.min_delta:
            self.count+=1
            if self.count >=self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


def train_DANN(X_source,Y_source,X_target,X_test,Y_test,
               feature_size=[10], label_size=[10],
               domain_size=[10],
               trade_off_adversarial='Step',
                lam_adversarial=1.0,
               batch_size= 32,lr = 0.0001,epochs = 200,
               print_freq=20,patience =10,min_delta = 0.01):

    X_source_tensor = torch.tensor(X_source, dtype=torch.float32)
    Y_source_tensor = torch.tensor(Y_source, dtype=torch.float32).view(-1, 1)
    X_target_tensor = torch.tensor(X_target, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1,1)

    #DataLoader
    source_dataset = TensorDataset(X_source_tensor, Y_source_tensor)
    target_dataset = TensorDataset(X_target_tensor,
                                   torch.zeros(len(X_target_tensor)))

    source_loader = DataLoader(source_dataset, batch_size=batch_size,
                               shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size,
                               shuffle=True)

    loss_fn = nn.BCELoss()

    in_feature = X_source_tensor.shape[1]
    model = AdversarialNet(in_feature= in_feature,
                           feature_size=feature_size,
                           label_size=label_size,
                           domain_size=domain_size,
                           max_iter=10000.0,
                        trade_off_adversarial=trade_off_adversarial,
                           lam_adversarial=lam_adversarial)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience,min_delta=min_delta)
    for epoch in range(epochs):
        total_correct_labels, total_samples_labels = 0, 0
        total_correct_domains, total_samples_domains = 0, 0
        target_iter = cycle(target_loader)

        for i, (source_X_batch, source_Y_batch) in enumerate(source_loader):
            target_X_batch, _ = next(target_iter)
            optimizer.zero_grad()

            #feature forward
            source_predict, source_domains = model(source_X_batch)
            source_label_loss = loss_fn(source_predict, source_Y_batch)
            source_domain_loss = loss_fn(source_domains,
                                      torch.ones(len(source_domains),
                                                 dtype=torch.float32).view(-1,1))

            _, target_domains = model(target_X_batch)
            target_domain_loss = loss_fn(target_domains,
                                      torch.zeros(len(target_domains),
                                                 dtype=torch.float32).view(-1,1))

            #total_loss
            losses = source_label_loss + source_domain_loss + target_domain_loss
            losses.backward()
            optimizer.step()

            #label_classifer acc
            correct_labels = ((source_predict >= 0.5).float() == source_Y_batch).sum().item()
            total_correct_labels += correct_labels
            total_samples_labels +=len(source_predict)
            domain_samles = len(source_domains)+len(target_domains)
            correct_domain = (((source_domains >= 0.5).float()).sum().item()+
                              ((target_domains <= 0.5).float()).sum().item())
            total_correct_domains += correct_domain
            total_samples_domains +=domain_samles

        epoch_acc_label = total_correct_labels / total_samples_labels * 100
        epoch_acc_domain = total_correct_domains / total_samples_domains * 100
        if (epoch + 1) % print_freq == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], "
                  f"Label Acc: {epoch_acc_label:.2f}%, "
                  f"Domain Acc: {epoch_acc_domain:.2f}%")

        model.eval()
        with torch.no_grad():
            test_pred, _ = model(X_test_tensor)
            test_loss_value = loss_fn(test_pred, Y_test_tensor).item()
            early_stopping(test_loss_value)

            test_correct = ((test_pred >= 0.5).float() == Y_test_tensor).sum().item()
            test_acc = test_correct / len(Y_test_tensor) * 100
            if (epoch + 1) % print_freq == 0:
                print(f"Epoch-Test [{epoch + 1}/{epochs}], "
                      f"Test Loss: {test_loss_value:.4f}, "
                      f"Test Accuracy: {test_acc:.2f}%")
        if early_stopping.early_stop:
            return model,test_acc,epoch_acc_domain
    print("Training Complete.")
    return model, test_acc, epoch_acc_domain


class FeatureLabelNet(nn.Module):
    def __init__(self, feature_extractor, label_classifier):
        super(FeatureLabelNet, self).__init__()
        self.feature_layers = nn.ModuleList(list(feature_extractor))
        self.label_layers = nn.ModuleList(list(label_classifier))

    def forward(self, x):
        for layer in self.feature_layers:
            x = layer(x)
        for layer in self.label_layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    X, Y, Xt, Yt = make_trans_moons(35, nb=300)
    Xt_train, Xt_test, Yt_train, Yt_test = train_test_split(Xt, Yt,
                                                            test_size=0.3,
                                                        random_state=42)
    model = train_DANN(X,Y,Xt_train,Xt_test,Yt_test,
                       feature_size=[20],
                       label_size=[],
                       domain_size=[],
                       epochs=300,
                       batch_size=64,lr=0.01,
                       trade_off_adversarial='Step',patience=40,min_delta=0.01)





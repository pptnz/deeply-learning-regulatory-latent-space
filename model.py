import os
import data_loader
import torch
import torch.nn as nn
import numpy as np
import csv
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_curve, auc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 20

num_kernel = 20
HM_vector_size = 40
ME_vector_size = 10
TF_vector_size = 5

reduced_HM_vector_size = 40
reduced_ME_vector_size = 10
reduced_TF_vector_size = 5

TF_self_attention_weight_size = 30
self_attention_weight_size = 50
num_rnn_layers = 1

torch.manual_seed(1)


class Attention(nn.Module):
    def __init__(self, sequence_length, hidden_size):
        super(Attention, self).__init__()

        self.sequence_length = sequence_length
        self.softmax = nn.Softmax(dim=1)
        self.W1 = nn.Linear(hidden_size, 16)
        self.w2 = nn.Linear(16, 1)

    def forward(self, x, dynamic=False, dynamic_length=None):
        # x: [batch_size, seq_length, hidden_size)

        alpha = self.w2(torch.tanh(self.W1(x))) # [batch_size, seq_length, 1]
        alpha.squeeze_(2) # [batch_size, seq_length]

        if dynamic:
            mask = torch.ones(alpha.size()).to(device)
            for i, l in enumerate(dynamic_length):
                if l < self.sequence_length:
                    mask[i, l:] = 0
            alpha *= mask

        alpha = self.softmax(alpha) # [batch_size, seq_length]

        context_vector = torch.bmm(alpha.unsqueeze(1), x) # [batch_size, 1, hidden_size]
        context_vector.squeeze_(1) # [batch_size, hidden_size]

        return context_vector, alpha


# CNN, LSTM, and attention for embedding histone marks
class HM_CRNN(nn.Module):
    def __init__(self):
        super(HM_CRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, num_kernel, kernel_size=(7, 7), stride=1, padding=0),
            nn.BatchNorm2d(num_kernel),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)

        self.rnn = nn.LSTM(input_size=num_kernel, hidden_size=HM_vector_size,
                           num_layers=num_rnn_layers, bidirectional=True)

        self.attention = Attention(sequence_length=12, hidden_size=2 * HM_vector_size)

    def forward(self, x):
        # 1. CNN
        # input x: [batch_size, 1, 7, 40]
        out = self.conv(x) # [batch_size, num_kernel, 1, 40-7+1]
        out.squeeze_(2) # [batch_size, num_kernel, 40-7+1]
        out = self.pool(out) # [batch_size, num_kernel, 12]

        # 2. LSTM
        out = out.permute(2, 0, 1) # [seq_length (12), batch_size, input_size]

        # RNN cell: [num_layers, batch_size, hidden_size]
        h0 = torch.zeros(num_rnn_layers * 2, out.size(1), HM_vector_size).to(device)
        c0 = torch.zeros(num_rnn_layers * 2, out.size(1), HM_vector_size).to(device)

        out, _ = self.rnn(out, (h0, c0)) # [seq_length, batch_size, hidden_size]

        # 3. Attention
        out = out.permute(1, 0, 2) # [batch_size, seq_length, hidden_size]
        context_vector, alpha = self.attention(out)

        return context_vector, alpha


# Dynamic LSTM and attention for embedding DNA methylation
class ME_RNN(nn.Module):
    def __init__(self):
        super(ME_RNN, self).__init__()

        self.rnn = nn.LSTM(input_size=2, hidden_size=ME_vector_size, num_layers=num_rnn_layers,
                           batch_first=True, bidirectional=True)
        self.attention = Attention(sequence_length=436, hidden_size=2 * ME_vector_size)

    def forward(self, ME, ME_length):
        # 1. Dynamic LSTM
        # input ME: [batch_size, sequence_length, input_size]
        pack = torch.nn.utils.rnn.pack_padded_sequence(ME, ME_length, batch_first=True, enforce_sorted=False)

        # cell: [num_layers * 2, batch_size, hidden_size]
        h0 = torch.zeros(num_rnn_layers * 2, ME.size(0), ME_vector_size).to(device)
        c0 = torch.zeros(num_rnn_layers * 2, ME.size(0), ME_vector_size).to(device)

        out, _ = self.rnn(pack, (h0, c0))

        # unpakced: [max_seq_length (436), batch_size, hidden_size (2 * ME_vector_size)]
        unpacked, length = torch.nn.utils.rnn.pad_packed_sequence(out, total_length=436)

        # 2. Attention
        unpacked = unpacked.transpose(0, 1) # [batch_size, max_seq_length (436), hidden_size (2 * ME_vector_size)]

        # context_vector: [batch_size, hidden_size (2 * ME_vector_size)]
        context_vector, alpha = self.attention(unpacked, dynamic=True, dynamic_length=length)

        return context_vector, alpha


# Self Attention Network for embedding transcription factors
class TF_SelfAttention(nn.Module):
    def __init__(self):
        super(TF_SelfAttention, self).__init__()

        self.W1 = nn.Linear(3, TF_self_attention_weight_size, bias=True)
        self.W2 = nn.Linear(3, TF_self_attention_weight_size, bias=True)

        self.softmax = nn.Softmax(dim=2)
        self.batch_norm = nn.BatchNorm1d(1016 * 3)
        self.fc = nn.Linear(1016 * 3, TF_vector_size)

    def forward(self, TF):
        TF = TF.transpose(1, 2) # [batch_size, input_size (1016), 3]

        Q = self.W1(TF) # [batch_size, input_size (1016), TF_self_attention_weight_size]
        K = self.W2(TF) # [batch_size, input_size (1016), TF_self_attention_weight_size]
        V = TF # [batch_size, input_size (1016), 3]

        QK = torch.bmm(Q, K.transpose(1, 2)) # [batch_size, input_size (1016), input_size (1016)]
        alpha_matrix = self.softmax(QK) # [batch_size, input_size (1016), input_size (1016)]

        out = torch.bmm(alpha_matrix, V) # [batch_size, input_size (1016), 3]
        out = out.view(out.size(0), -1) # [batch_size, input_size (1016) * 3]
        out = self.batch_norm(out) # [batch_size, input_size (1016), 3]
        out = self.fc(out) # [batch_size, TF_vector_size]

        return torch.tanh(out)


# Multi-attention Block from Multi-attention Recurrent Network for integrating context vectors
# Paper: https://arxiv.org/pdf/1802.00923.pdf
class MultiAttention(nn.Module):
    def __init__(self):
        super(MultiAttention, self).__init__()
        self.num_attentions = 10
        self.softmax = nn.Softmax(dim=2)

        self.A = nn.Linear(2 * HM_vector_size + 2 * ME_vector_size + TF_vector_size,
                            (2 * HM_vector_size + 2 * ME_vector_size + TF_vector_size) * self.num_attentions)

        self.HM_dim_reduce_net = nn.Linear(self.num_attentions * 2 * HM_vector_size, 2 * reduced_HM_vector_size)
        self.ME_dim_reduce_net = nn.Linear(self.num_attentions * 2 * ME_vector_size, 2 * reduced_ME_vector_size)
        self.TF_dim_reduce_net = nn.Linear(self.num_attentions * TF_vector_size, reduced_TF_vector_size)

        self.dim_reduce_nets = [self.HM_dim_reduce_net, self.ME_dim_reduce_net, self.TF_dim_reduce_net]

    def forward(self, context_vector_list):
        context_vectors = torch.cat(context_vector_list, dim=1) # [batch_size, 2 * HM_vector_size + 2 * ME_vector_size + TF_vector_size]

        multi_attention_out = self.A(context_vectors) # [batch_size, (2 * HM_vector_size + 2 * ME_vector_size + TF_vector_size) * num_attentions]
        multi_attention_out = multi_attention_out.reshape([multi_attention_out.size(0), self.num_attentions, -1]) # [batch_size, num_attentions, 2 * HM_vector_size + 2 * ME_vector_size + TF_vector_size]

        alpha = self.softmax(multi_attention_out) # [batch_size, num_attentions, 2 * HM_vector_size + 2 * ME_vector_size + TF_vector_size]
        alpha = alpha.reshape([multi_attention_out.size(0), -1]) # [batch_size, (2 * HM_vector_size + 2 * ME_vector_size + TF_vector_size) * num_attentions]

        expanded_h = torch.cat([vector.repeat(1, self.num_attentions) for vector in context_vector_list], dim=1) # [batch_size, (2 * HM_vector_size + 2 * ME_vector_size + TF_vector_size) * num_attentions]

        # Element-wise multiplication
        h_tilde = alpha * expanded_h

        # Reduce dimensionality
        start = 0
        out = []
        for i, vector in enumerate(context_vector_list):
            vector_size = vector.shape[1] * self.num_attentions
            dim_reduced = self.dim_reduce_nets[i](h_tilde[:, start: start + vector_size])
            out.append(dim_reduced)
            start += vector_size

        output = torch.cat(out, dim=1) # [batch_size, 2 * reduced_HM_vector_size + 2 * reduced_ME_vector_size + reduced_TF_vector_size]
        return output, alpha


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.hm_crnn = HM_CRNN()
        self.me_rnn = ME_RNN()
        self.tf_self_attention = TF_SelfAttention()

        self.mab = MultiAttention()
        self.fc1 = nn.Linear(2 * reduced_HM_vector_size + 2 * reduced_ME_vector_size + reduced_TF_vector_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, HM, ME, ME_length, TF):
        HM_vector, HM_alpha = self.hm_crnn(HM)
        ME_vector, ME_alpha = self.me_rnn(ME, ME_length)
        TF_vector = self.tf_self_attention(TF)

        HM_vector.squeeze_()
        ME_vector.squeeze_()

        context_vector_list = [HM_vector, ME_vector, TF_vector]
        out, alpha = self.mab(context_vector_list)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out.squeeze_()

        return torch.sigmoid(out), alpha, HM_alpha, ME_alpha


def test(model, data_loader, save_alphas=False):
    model.eval()
    if save_alphas:
        alphas = []
        HM_alphas = []
        ME_alphas = []

    with torch.no_grad():
        GEs = []
        GE_values = []
        pred = []
        correct = 0
        total = 0
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for i, (gene_id, HM, ME, ME_length, TF, GE, GE_value) in enumerate(data_loader):
            HM = HM.to(device)
            ME = ME.to(device)
            ME_length = ME_length.to(device)
            TF = TF.to(device)
            GE = GE.to(device)

            output, alpha, HM_alpha, ME_alpha = model(HM, ME, ME_length, TF)

            if save_alphas:
                alphas.append(alpha)
                HM_alphas.append(HM_alpha)
                ME_alphas.append(ME_alpha)

            total += GE.size(0)
            correct += (torch.round(output) == GE.float()).sum().item()

            true_positive += ((torch.round(output) == GE.float()).cpu().detach().numpy()
                              & (GE.float() == 1).cpu().detach().numpy()).sum().item()
            true_negative += ((torch.round(output) == GE.float()).cpu().detach().numpy()
                              & (GE.float() == 0).cpu().detach().numpy()).sum().item()
            false_positive += ((torch.round(output) != GE.float()).cpu().detach().numpy()
                              & (GE.float() == 0).cpu().detach().numpy()).sum().item()
            false_negative += ((torch.round(output) != GE.float()).cpu().detach().numpy()
                              & (GE.float() == 1).cpu().detach().numpy()).sum().item()

            GEs.append(GE.float())
            GE_values.append(GE_value)
            pred.append(output)

        # Correlation
        GE_tensor = torch.cat(GEs, dim=0)
        GE_value_tensor = torch.cat(GE_values, dim=0)
        pred_tensor = torch.cat(pred, dim=0)
        GE_value_tensor.unsqueeze_(1)
        pred_tensor.unsqueeze_(1)

        rank_corr, p = kendalltau(GE_value_tensor.cpu().detach().numpy(), pred_tensor.cpu().detach().numpy())

    accuracy = 100 * correct / total

    fpr, tpr, thresholds = roc_curve(GE_tensor.cpu().detach().numpy(), pred_tensor.cpu().detach().numpy())
    avgAUC = auc(fpr, tpr)

    if save_alphas:
        alpha_tensor = torch.cat(alphas, dim=0)

        HM_alpha_tensor = torch.cat(HM_alphas, dim=0)
        ME_alpha_tensor = torch.cat(ME_alphas, dim=0)

        np.savetxt('alpha_tensor.csv', alpha_tensor.cpu().detach().numpy(), delimiter=',')
        np.savetxt('HM_alpha_tensor.csv', HM_alpha_tensor.cpu().detach().numpy(), delimiter=',')
        np.savetxt('ME_alpha_tensor.csv', ME_alpha_tensor.cpu().detach().numpy(), delimiter=',')

    return accuracy, avgAUC, rank_corr, true_positive, true_negative, false_positive, false_negative


def focal_loss(output, target):
    gamma = 2
    loss = target * torch.log(output) * torch.pow(1 - output, gamma) + (1 - target) * torch.log(1 - output) * torch.pow(output, gamma)
    loss = -torch.sum(loss)

    return loss


cell_lines = ["E003"]
for cell_line in cell_lines:
    # 4-fold cross validation
    for seed in range(4):
        print("cell line:", cell_line, "seed:", seed)
        train_loader, val_loader, test_loader = data_loader.make_data_loader(cell_line, seed)

        model = Model().to(device)

        if os.path.isfile(cell_line + '_' + str(seed) + '.ckpt'):
            model.load_state_dict(torch.load(cell_line + '_' + str(seed) + '.ckpt'))
            print("model loaded")

        # criterion = nn.BCELoss()
        criterion = focal_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_AUC = 0
        best_test_AUC = 0
        best_val_rank_corr = 0
        best_test_rank_corr = 0

        best_val_TP = 0
        best_test_TP = 0
        best_val_TN = 0
        best_test_TN = 0

        best_val_FP = 0
        best_test_FP = 0
        best_val_FN = 0
        best_test_FN = 0

        total_step = len(train_loader)

        log = csv.writer(open(cell_line + '_' + str(seed) + ".log", 'w'))
        log.writerow(["epoch", "Val AUC", "Test AUC", "Val Corr", "Test Corr",
                      "Val TP", "Test TP", "Val TN", "Test TN",
                      "Val FP", "Test FP", "Val FN", "Test FN"])

        for epoch in range(num_epochs):
            model.train()
            for i, (gene_id, HM, ME, ME_length, TF, GE, _) in enumerate(train_loader):
                HM = HM.to(device)
                ME = ME.to(device)
                ME_length = ME_length.to(device)
                TF = TF.to(device)
                GE = GE.to(device)

                output, alpha, HM_alpha, ME_alpha = model(HM, ME, ME_length, TF)
                loss = criterion(output, GE.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                             loss.item()))

            # Validate the model
            val_accuracy, val_AUC, val_rank_corr, val_TP, val_TN, val_FP, val_FN = test(model, val_loader)
            print('Validation AUC:\t{} %'.format(100 * val_AUC))
            print('Validation Rank Concordance:\t{} %\n'.format(100 * val_rank_corr))

            # Test the model
            test_accuracy, test_AUC, test_rank_corr, test_TP, test_TN, test_FP, test_FN = test(model, test_loader)
            print('Test AUC:\t\t{} %'.format(100 * test_AUC))
            print('Test Rank Concordance:\t{} %\n'.format(100 * test_rank_corr))
            log.writerow([epoch, 100 * val_AUC, 100 * test_AUC, 100 * val_rank_corr, 100 * test_rank_corr,
                          val_TP, test_TP, val_TN, test_TN,
                          val_FP, test_FP, val_FN, test_FN])

            if val_AUC > best_val_AUC:
                best_val_AUC = val_AUC
                best_test_AUC = test_AUC

                best_val_rank_corr = val_rank_corr
                best_test_rank_corr = test_rank_corr

                best_val_TP = val_TP
                best_test_TP = test_TP
                best_val_TN = val_TN
                best_test_TN = test_TN

                best_val_FP = val_FP
                best_test_FP = test_FP
                best_val_FN = val_FN
                best_test_FN = test_FN

                torch.save(model.state_dict(), cell_line + '_' + str(seed) + '.ckpt')

        print('Best Model Validation AUC:\t{} %'.format(100 * best_val_AUC))
        print('Best Model Test AUC:\t\t{} %\n'.format(100 * best_test_AUC))

        log.writerow(["best", 100 * best_val_AUC, 100 * best_test_AUC, 100 * best_val_rank_corr, 100 * best_test_rank_corr,
                      best_val_TP, best_test_TP, best_val_TN, best_test_TN,
                      best_val_FP, best_test_FP, best_val_FN, best_test_FN])

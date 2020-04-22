import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_loss(model, feature, label, batch, opt, loss_func, train, device): 
    total_loss=0
    start=0
    for j in range(int(len(feature)/batch)+1):
        if start+batch>len(feature):
            end=-1
        else:
            end=start+batch
            
        x=feature[start:end].to(device) #x,y放到gpu裡
        y=label[start:end].to(device)
        out=model(x)
        out=out.to(device)
        loss=loss_func(out,y)
        if train:
            opt.zero_grad() #梯度歸零
            loss.backward() #back propogation
            opt.step() #更新參數
        #print('batch loss:'+str(loss.item()))
        total_loss+=loss.item()
        torch.cuda.empty_cache()
        start=end
    total_loss=total_loss/(int(len(feature)/batch)+1)
    return total_loss

def mape(prediction,truth,M,m):
    diff=(prediction-truth).detach().numpy()*M
    diff=np.absolute(diff/(truth.detach().numpy()*M+m))
    return diff.mean()


def mae(prediction, truth):
    diff=np.absolute((prediction-truth).detach().numpy())
    return diff.mean()


def plot_result(model, xs, ys, output_len, title, target_series, scale, save_path):
    result=[]
    truth=[]
    for i in range(2):
        for j in range(0,xs[i].shape[0],output_len):
            prediction=model(xs[i][j].unsqueeze(0))[:,:,0].squeeze(0).tolist()
            y=ys[i][j,:,0].tolist()
        
            result.extend([k*(scale[0][target_series]-scale[1][target_series])+scale[1][target_series] for k in prediction])
            truth.extend([k*(scale[0][target_series]-scale[1][target_series])+scale[1][target_series] for k in y])
        
    plt.plot(truth)
    plt.plot(result,alpha=0.5)
    plt.legend(['truth','prediction'])
    plt.title(title)
    plt.savefig(save_path+title+'_prediction.png')
    plt.show()
    plt.clf()
    
    return result,truth
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def plot_attention(model, X_train, X_test, column_names, neighbors, save_path):
    weights=[]
    for i in range(len(X_train)):
        _,_,sample=model.encoder(X_train[i].unsqueeze(0))
        weights.append(sample[0][0].tolist())
    
    for i in range(len(X_test)):
        _,_,sample=model.encoder(X_test[i].unsqueeze(0))
        weights.append(sample[0][0].tolist())    
        
    w=np.array(weights).transpose()
    #w.shape
    plt.figure(figsize=[8,5])
    plt.pcolor(w)
    plt.yticks([i for i in range(len(column_names)-1)],[column_names[i] for i in neighbors[1:]])
    plt.title(column_names[neighbors[0]])

    plt.savefig(save_path+column_names[neighbors[0]]+'_attention.png')
    plt.show()
    plt.clf()
    
    return weights
    